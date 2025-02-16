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
                bat 'start /B venv\\Scripts\\python app.py > flask_app.log 2>&1'  // Run Flask app in background and log output
                // Capture the PID of Flask app
                script {
                    def pid = bat(script: 'tasklist /FI "IMAGENAME eq python.exe" /NH', returnStdout: true).trim()
                    echo "Flask app PID: ${pid}"
                }
            }
        }

        // Stage to stop the Flask app using PID (you need to capture the specific Flask app PID)
        stage('Stop Flask App') {
            steps {
                echo 'Stopping Flask app'
                bat 'taskkill /F /PID ${pid}'  // Kill Flask app using PID (modify according to how you capture PID)
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
