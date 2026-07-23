#!/bin/bash
echo "Setting up Resume Screener..."
pip install -r requirements.txt
python -m spacy download en_core_web_sm
echo "Setup complete. Run: uvicorn main:app --reload --port 8010"
