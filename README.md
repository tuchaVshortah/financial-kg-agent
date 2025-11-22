# Step A — Implement core OOP classes

- FinancialLLM
- FinancialKG
- DataRetriever
- LLMController
- These will form the backbone of your agent.

# Step B — Implement KG builder

- Programmatically generate a small RDF graph
- Save as .ttl
- Include basic entities: Client, Transaction, ComplianceRule, etc.

# Step C — Implement KG query engine

- Basic SPARQL wrapper using rdflib
- Helper functions to extract structured facts

# Step D — Implement LLM orchestration

- Prompt templates
- Fact-injection logic
- Retrieval loop

# Step E — Build the demonstration Colab notebook

- Load the repo
- Build KG
- Ask agent a few queries
- Show evaluation examples
