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

        stage('Build the Flask App') {
            steps {
                script {
                    echo 'Building the Flask app'
                    // Start the Flask app in the background
                    bat 'start /B venv\\Scripts\\python app.py'  // Run Flask app in background

                    // Wait for a short time (e.g., 10 seconds) to let the app start
                    sleep(10)

                    // Kill the Flask app by terminating the Python process (this kills all python processes running, be cautious!)
                    bat 'taskkill /F /IM python.exe'
                    
                    echo 'App started and killed successfully'
                }
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
