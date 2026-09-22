import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Create a sample transaction dataset
dataset = [
    ['milk', 'bread', 'butter'],
    ['bread', 'butter'],
    ['milk', 'butter'],
    ['milk', 'bread', 'butter', 'jam'],
    ['bread', 'jam']
]

# Print each transaction on a separate line
for transaction in dataset:
    print(transaction)

# convert dataset into a DataFrame (One-Hot Encoding format)
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori Algorithm
frequent_items = apriori(df, min_support=0.6, use_colnames=True)

# generate association rules
rules = association_rules(frequent_items, metric='lift', min_threshold=1.0)

# Display results
print("Frequent Itemsets:")
print(frequent_items)
print('\n Assocaition Rules:')
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
