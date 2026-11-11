# ATLAS — Graph Intelligence Explorer

Day 7 of the 60-Day End-of-Year Project Marathon.

ATLAS is an interactive graph analytics application that transforms
relationship data into an explorable network.

## Features

- Generate a built-in sample network.
- Upload custom CSV edge lists.
- Visualise entities and their relationships.
- Detect communities using Greedy Modularity.
- Calculate Degree Centrality, Betweenness Centrality and PageRank.
- Investigate individual entities and their neighbours.
- Find shortest paths between two entities.
- Export node metrics, connections and community assignments.

## Technology Stack

- Python
- Streamlit
- NetworkX
- Pandas
- NumPy
- Plotly

## Dataset Format

Your CSV should contain at least two columns:

| source | target | weight |
|---|---|---|
| User A | User B | 2.5 |
| User B | User C | 1.0 |
| User A | User D | 3.2 |

The weight column is optional.

The built-in demo network can be used without downloading a dataset.

## Installation — Windows / PowerShell

Open the project folder in VS Code.

Create a virtual environment:

    py -3.10 -m venv .venv

Activate it:

    .\.venv\Scripts\Activate.ps1

If PowerShell blocks activation:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install --upgrade pip --timeout 600 --retries 10

    python -m pip install -r requirements.txt --no-cache-dir --timeout 600 --retries 10

Run the application:

    streamlit run app.py

## Analytical Methods

- Degree Centrality: measures direct connectivity.
- Betweenness Centrality: identifies potential bridge entities.
- PageRank: estimates node importance based on network connections.
- Greedy Modularity: detects groups of densely connected entities.
- Shortest Path: finds the minimum-hop route between two entities.

## Important Note

This project uses classical graph algorithms. Its findings describe
the uploaded network structure and should not automatically be
interpreted as proof of real-world influence, causation or risk.

The shortest-path feature uses an unweighted minimum-hop path.
