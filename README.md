# Main Development Objectives

## Step A — Implement core OOP classes

| Component     | Status |
| ------------- | ------ |
| FinancialLLM  | ✅     |
| FinancialKG   | ✅     |
| DataRetriever | ✅     |
| LLMController | ✅     |

## Step B — Implement KG builder

- Programmatically generate a small RDF graph
- Save as .ttl
- Include basic entities: Client, Transaction, ComplianceRule, etc.

## Step C — Implement KG query engine

- Basic SPARQL wrapper using rdflib
- Helper functions to extract structured facts

## Step D — Implement LLM orchestration

- Prompt templates
- Fact-injection logic
- Retrieval loop

## Step E — Build the demonstration Colab notebook

- Load the repo
- Build KG
- Ask agent a few queries
- Show evaluation examples

# Miscellaneous stuff

## Todos

- Fix/improve LLM reasoning
- Add more KG data & rule definitions
- Extend retriever with more functions
- Add JSON evaluation output
- Add a memory system (RAP-lite buffer)
- Something else

## Rules for the LLM

- The LLM must never infer compliance from incomplete data.
- The LLM must only answer “not available / unknown” if a field is missing.
- he LLM must treat conflicting facts as “inconclusive”.
