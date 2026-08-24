pipeline {
  agent any

  environment {
    IMAGE = 'ghcr.io/360basirah/360basirah-ocr'
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Build & Push') {
      steps {
        withCredentials([usernamePassword(
          credentialsId: '360Basirah',
          usernameVariable: 'GH_USER',
          passwordVariable: 'GH_TOKEN'
        )]) {
          sh '''
            echo "$GH_TOKEN" | docker login ghcr.io -u "$GH_USER" --password-stdin
            docker build -t $IMAGE:latest -t $IMAGE:build-$BUILD_NUMBER .
            docker push $IMAGE:latest
            docker push $IMAGE:build-$BUILD_NUMBER
          '''
        }
      }
    }
  }

  post {
    always {
      sh 'docker image prune -f || true'
    }
  }
}
