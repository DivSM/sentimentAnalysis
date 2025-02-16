pipeline {
    agent any

    environment {
        // Set environment variables
        REPO_URL = 'https://github.com/DivSM/sentimentAnalysis.git' 
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out repository'
                git branch: 'master', url: "${REPO_URL}"
            }
        }

        stage('Setup and Install Dependencies') {
            steps {
                echo 'Setting up virtual environment and installing dependencies'
                bat 'python -m venv venv'  // Create virtual environment
                bat 'venv\\Scripts\\pip install -r requirements.txt'  // Install dependencies from requirements.txt
            }
        }

        stage('Run Flask App') {
            steps {
                echo 'Running Flask app'
                bat 'venv\\Scripts\\python app.py'  // Run the app using the virtual environment's Python
            }
        }
    }

    post {
        success {
            echo 'Build and deployment successful!'
        }
        failure {
            echo 'Build or deployment failed!'
        }
    }
}
