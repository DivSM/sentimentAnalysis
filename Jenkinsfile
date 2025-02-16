pipeline {
    agent any

    environment {
        // Set environment variables
        VENV_PATH = 'venv'
        REPO_URL = 'https://github.com/DivSM/sentimentAnalysis.git' 
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out repository'
                // Checkout the repository
                git branch: 'master', url: "${REPO_URL}"
            }
        }

        stage('Setup Virtual Environment') {
            steps {
                script {
                    echo 'Creating a virtual environment'
                    // Create a virtual environment
                    sh 'python3 -m venv ${VENV_PATH}'
                    sh '${VENV_PATH}/bin/pip install --upgrade pip'
                    sh '${VENV_PATH}/bin/pip install -r requirements.txt'
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    
                    echo 'Testing'
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    echo 'Running app'
                    // Run Flask app
                    sh '${VENV_PATH}/bin/python app.py'
                }
            }
        }
    }

    post {
        success {
            echo 'Build and Deployment Successful!'
        }
        failure {
            echo 'Build or Deployment Failed!'
        }
    }
}
