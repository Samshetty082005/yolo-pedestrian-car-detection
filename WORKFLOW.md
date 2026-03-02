# Git Workflow - NO CONFLICTS GUIDE

## File Ownership (strict - never cross these lines!)

| File/Folder | Owner |
|---|---|
| src/train.py | Copilot |
| src/detect.py | Copilot |
| config/model_config.yaml | Copilot |
| requirements.txt | Copilot |
| dataset/ | Antigravity |
| config/dataset.yaml | Antigravity |
| src/evaluate.py | Antigravity |
| src/visualize.py | Antigravity |
| report/ | Antigravity |

## Order of Operations
1. Antigravity creates dataset branch → pushes → merges to main FIRST
2. Copilot pulls main → creates model branch → trains → pushes
3. Copilot merges to main
4. Antigravity pulls main → runs full test pipeline → generates report
5. Both review final output

## PR Rules
- Each person reviews the other's PR before merge
- Never force push to main
- Always pull before pushing