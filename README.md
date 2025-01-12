# Systems Thinking Analysis Project

This project implements a hierarchical model for analyzing systems thinking in corporate annual reports using advanced NLP techniques.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AWS credentials:
```bash
aws configure
```

## Usage

Run analysis:
```bash
python start.py path/to/document.pdf
```

## Project Structure

```
systems_thinking/
├── config/          # Configuration files
├── data/           # Data storage
├── src/            # Source code
└── tests/          # Unit tests
```