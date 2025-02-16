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

        stage('Find and Kill Flask App') {
            steps {
                script {
                    // Find the process ID (PID) of the Flask app
                    def pid = bat(script: 'for /f "tokens=5" %%a in (\'tasklist /fi "imagename eq python.exe" /fo csv\') do @echo %%a', returnStdout: true).trim()
                    
                    // Check if a process was found and kill it
                    if (pid) {
                        echo "Found Flask app running with PID: ${pid}. Terminating the process."
                        bat "taskkill /F /PID ${pid}"
                    } else {
                        echo "No Flask app process found."
                    }
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
