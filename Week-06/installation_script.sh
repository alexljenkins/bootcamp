"""
This .sh file can be run on an AWS machine for setting up the machine
Installation script. Run with ./, bash or source.

"""
sudo yum install -y python3
sudo yum install -y git
touch hello.py
echo "print('Hello from the AWS machine!\nEverything looks good!')" > hello.py
python3 hello.py

# git (clone an entire repository)
# boto3 (python library for AWS API)
# fabric (python library for running scripts across multiple machines)
