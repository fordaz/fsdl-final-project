echo "Pushing model $1"

# activating the aws instance conda environment ready for DL
conda activate pytorch_latest_p37

dvc add model_artifacts/$1
git add model_artifacts/$1.dvc model_artifacts/.gitignore

dvc add model_artifacts/$2
git add model_artifacts/$2.dvc model_artifacts/.gitignore 

git commit -m "Auto adding dvc artifacts for model $1"
git push origin main

dvc push