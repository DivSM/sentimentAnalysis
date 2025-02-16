pipeline {
    agent any

    environment {
        // Set environment variables
        VENV_PATH = 'myenv'
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
                    sh 'python3 -m myenv ${VENV_PATH}'
                    sh '${VENV_PATH}/Scripts/pip install --upgrade pip' // Windows specific path
                    sh '${VENV_PATH}/Scripts/pip install -r requirements.txt' // Windows specific path
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    echo 'Testing'
                    // Add test commands here if needed
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    echo 'Running app'
                    // Run Flask app (not using nohup or background execution)
                    sh '${VENV_PATH}/Scripts/python app.py' // Windows specific path
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
