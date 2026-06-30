"""
Script final para construir o dataset de 50 perguntas válidas.
Combina ragas_testset_valid.csv (21 perguntas antigas curadas) 
e ragas_testset_novas_50.csv (28 perguntas novas geradas)
+ 1 pergunta reaproveitada do dataset original com ground_truth corrigido.
"""
import pandas as pd

# Carrega datasets
valid_df = pd.read_csv('ragas_testset_valid.csv')
new_df = pd.read_csv('ragas_testset_novas_50.csv')
old_df = pd.read_csv('ragas_testset.csv')

# Filtra perguntas validas
def is_valid(row):
    gt = str(row['ground_truth']).strip().lower()
    q = str(row['question']).strip()
    return ('not present in context' not in gt) and len(q) > 10 and q != 'nan'

valid_questions = valid_df[valid_df.apply(is_valid, axis=1)].copy()
new_questions = new_df[new_df.apply(is_valid, axis=1)].copy()

print(f"VALID: {len(valid_questions)} perguntas")
print(f"NEW: {len(new_questions)} perguntas")
print(f"Subtotal: {len(valid_questions) + len(new_questions)}")

# Precisamos de mais 1 pergunta para chegar a 50
# Reaproveitamos "What is the purpose of the VertexLocationHint class in the Tez API?"
# com um ground_truth corrigido baseado no contexto do codigo fonte
extra_row = old_df[old_df['question'].str.contains('VertexLocationHint', na=False)].copy()
extra_row['ground_truth'] = (
    "The VertexLocationHint class in the Tez API is used to provide location hints "
    "for scheduling tasks within a vertex of a DAG, allowing the framework to make "
    "data-locality-aware scheduling decisions."
)

# Combina tudo
final_dataset = pd.concat([valid_questions, new_questions, extra_row], ignore_index=True)

# Garante colunas corretas
expected_cols = ['question', 'contexts', 'ground_truth', 'evolution_type', 'metadata', 'episode_done']
final_dataset = final_dataset[expected_cols]

print(f"\nDataset final: {len(final_dataset)} perguntas")
print(f"\nDistribuicao por evolution_type:")
print(final_dataset['evolution_type'].value_counts().to_string())

print(f"\nLista de perguntas:")
for i, row in final_dataset.iterrows():
    q = str(row['question'])[:95]
    et = str(row['evolution_type'])
    print(f"  {i+1:2d}. [{et:15s}] {q}")

# Salva
output_file = 'ragas_testset_final_50.csv'
final_dataset.to_csv(output_file, index=False)
print(f"\nDataset final salvo em: {output_file}")
print(f"Total: {len(final_dataset)} perguntas validas")
