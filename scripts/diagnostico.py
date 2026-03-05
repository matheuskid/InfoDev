from pymongo import MongoClient
import pandas as pd
from tqdm import tqdm

# CONEXÃO
client = MongoClient("mongodb://localhost:27017/")
db = client["smartshark_2_1"]

def analyze_projects():
    print("🔍 Iniciando escaneamento dos projetos (Isso pode demorar uns minutos)...")
    
    projects = list(db.project.find({}, {"name": 1, "_id": 1}))
    results = []

    for proj in tqdm(projects, desc="Analisando"):
        stats = {
            "Project": proj["name"],
            "Commits": 0,
            "Linked_Commits": 0,
            "Issues": 0,
            "Link_Rate": 0.0
        }

        # 1. Pega IDs dos Sistemas (Git e Jira)
        vcs = db.vcs_system.find_one({"project_id": proj["_id"]})
        issue_sys = db.issue_system.find_one({"project_id": proj["_id"]})

        # 2. Analisa Commits (Cooperação)
        if vcs:
            # Total de commits
            stats["Commits"] = db.commit.count_documents({"vcs_system_id": vcs["_id"]})
            
            # Commits que têm link com Issues (Rastreabilidade)
            # O campo 'linked_issue_ids' deve existir e não ser vazio array vazia
            stats["Linked_Commits"] = db.commit.count_documents({
                "vcs_system_id": vcs["_id"],
                "linked_issue_ids": {"$exists": True, "$ne": []}
            })

        # 3. Analisa Issues (Coordenação)
        if issue_sys:
            stats["Issues"] = db.issue.count_documents({"issue_system_id": issue_sys["_id"]})

        # 4. Calcula Taxa de Qualidade
        if stats["Commits"] > 0:
            stats["Link_Rate"] = round((stats["Linked_Commits"] / stats["Commits"]) * 100, 2)

        results.append(stats)

    # Cria DataFrame e ordena pelos melhores
    df = pd.DataFrame(results)
    
    # Filtra projetos vazios ou muito ruins (menos de 100 commits)
    df_clean = df[df["Commits"] > 100].sort_values(by="Link_Rate", ascending=False)

    print("\n🏆 RANKING DE PROJETOS PARA SEU TCC 🏆")
    print(df_clean.to_markdown(index=False))
    
    # Salva em CSV para você consultar depois
    df_clean.to_csv("smartshark_diagnosis.csv", index=False)
    print("\n✅ Relatório salvo em 'smartshark_diagnosis.csv'")

if __name__ == "__main__":
    analyze_projects()