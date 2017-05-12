rm -f assignment3.zip
zip -r assignment3.zip . -x "*.git" "*cs294_129/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements.txt" ".env/*" "*.pyc" "*cs231n/build/*"
