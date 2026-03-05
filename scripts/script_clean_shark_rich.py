from pymongo import MongoClient
from tqdm import tqdm
import sys

# ==============================================================================
# CONFIGURAÇÃO: LISTA DE PROJETOS
# ==============================================================================
# Coloque aqui os nomes EXATOS como estão no SmartSHARK
TARGET_PROJECTS = [
    "tez", 
    "pdfbox", 
    "nifi", 
    "ranger", 
    "phoenix" 
]

SOURCE_DB = "smartshark_2_1"
TARGET_DB = "clean_shark"

# ==============================================================================
# PREPARAÇÃO
# ==============================================================================
client = MongoClient("mongodb://localhost:27017/")
db_in = client[SOURCE_DB]
db_out = client[TARGET_DB]

def setup_indexes():
    print("⚡ Garantindo índices para performance...")
    # Índices no banco de origem (leitura)
    db_in.file_action.create_index("commit_id")
    db_in.hunk.create_index("file_action_id")
    db_in.file.create_index("_id")
    # Índices no banco de destino (escrita/busca futura)
    db_out.rich_commits.create_index("hash")
    db_out.rich_issues.create_index("original_id")

def clean_target_db():
    print("🧹 Limpando o banco de destino (uma única vez)...")
    db_out.rich_issues.drop()
    db_out.rich_emails.drop()
    db_out.rich_commits.drop()
    print("✅ Banco limpo. Iniciando carga cumulativa.")

def process_project(project_name):
    print(f"\n==================================================")
    print(f"🚀 PROCESSANDO PROJETO: {project_name}")
    print(f"==================================================")

    # 1. PEGAR ID DO PROJETO
    project = db_in.project.find_one({"name": project_name})
    if not project:
        print(f"❌ ERRO: Projeto '{project_name}' não encontrado! Pulando...")
        return
    
    proj_id = project["_id"]

    # ==========================================================
    # A. ISSUES (Coordenação)
    # ==========================================================
    issue_system = db_in.issue_system.find_one({"project_id": proj_id})
    if issue_system:
        cursor = db_in.issue.find({"issue_system_id": issue_system["_id"]}).batch_size(1000)
        buffer = []
        
        for issue in tqdm(cursor, desc=f"[{project_name}] Issues"):
            # Pega comentários
            comments_cursor = db_in.issue_comment.find({"issue_id": issue["_id"]})
            comments_text = [c.get("comment", "") for c in comments_cursor]
            
            # Contexto do Projeto no Texto
            full_text = f"Project: {project_name}\nType: Issue\nTitle: {issue.get('title')}\nDescription: {issue.get('desc')}\n"
            if comments_text:
                full_text += "Comments:\n" + "\n---\n".join(comments_text)

            doc = {
                "project": project_name, # Campo importante para filtrar depois
                "type": "issue",
                "original_id": issue["_id"],
                "title": issue.get("title"),
                "status": issue.get("status"),
                "text_for_embedding": full_text,
                "created_at": issue.get("created_at")
            }
            buffer.append(doc)
            if len(buffer) >= 1000:
                db_out.rich_issues.insert_many(buffer)
                buffer = []
        
        if buffer: db_out.rich_issues.insert_many(buffer)

    # ==========================================================
    # B. EMAILS (Comunicação)
    # ==========================================================
    mailing_lists = list(db_in.mailing_list.find({"project_id": proj_id}))
    ml_ids = [ml["_id"] for ml in mailing_lists]

    if ml_ids:
        cursor = db_in.message.find({"mailing_list_id": {"$in": ml_ids}}).batch_size(1000)
        buffer = []

        for msg in tqdm(cursor, desc=f"[{project_name}] Emails"):
            doc = {
                "project": project_name,
                "type": "email",
                "subject": msg.get("subject"),
                "text_for_embedding": f"Project: {project_name}\nType: Email\nSubject: {msg.get('subject')}\nBody: {msg.get('body')}",
                "date": msg.get("date")
            }
            buffer.append(doc)
            if len(buffer) >= 1000:
                db_out.rich_emails.insert_many(buffer)
                buffer = []
        
        if buffer: db_out.rich_emails.insert_many(buffer)

    # ==========================================================
    # C. COMMITS + CÓDIGO (Cooperação)
    # ==========================================================
    vcs = db_in.vcs_system.find_one({"project_id": proj_id})
    if vcs:
        cursor = db_in.commit.find({"vcs_system_id": vcs["_id"]}, no_cursor_timeout=True).batch_size(20)
        buffer = []
        
        for commit in tqdm(cursor, desc=f"[{project_name}] Commits"):
            
            # Filtro de Sanidade: Pula commits gigantes (>100 arquivos)
            file_count = db_in.file_action.count_documents({"commit_id": commit["_id"]})
            if file_count > 100: continue 

            commit_hash = commit.get('revision_hash', 'unknown')
            
            # Cabeçalho com Contexto do Projeto
            header = (
                f"Project: {project_name}\n"
                f"Type: Commit Code Change\n"
                f"Message: {commit.get('message', '') or 'No message'}\n"
                f"Date: {commit.get('author_date')}\n"
                f"Hash: {commit_hash}\n"
            )
            
            # Extração de Código (Lógica Full sem Limit)
            file_actions = db_in.file_action.find({"commit_id": commit["_id"]})
            full_diff = ""
            files_list = []
            
            for fa in file_actions:
                # Tenta pegar nome do arquivo
                f_path = "unknown"
                try:
                    f_doc = db_in.file.find_one({"_id": fa["file_id"]})
                    if f_doc: f_path = f_doc.get("path", "unknown")
                except: pass
                files_list.append(f_path)
                
                # Pega Hunks
                hunks = db_in.hunk.find({"file_action_id": fa["_id"]})
                for h in hunks:
                    content = h.get("content", "")
                    if len(content) < 5000: # Proteção contra arquivo minificado
                        full_diff += f"\n--- File: {f_path} ---\n{content}"

            doc = {
                "project": project_name,
                "type": "commit",
                "hash": commit_hash,
                "date": commit.get("author_date"),
                "text_for_embedding": header + "\nCHANGES:\n" + full_diff,
                "files_touched": files_list
            }
            buffer.append(doc)
            
            if len(buffer) >= 50: 
                db_out.rich_commits.insert_many(buffer)
                buffer = []
        
        if buffer: db_out.rich_commits.insert_many(buffer)

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    setup_indexes()     # Cria índices 1 vez
    clean_target_db()   # Limpa o destino 1 vez
    
    # Loop Principal: Itera sobre a lista de projetos
    for proj in TARGET_PROJECTS:
        process_project(proj)
        
    print("\n✅ Processamento Multi-Projeto Concluído!")