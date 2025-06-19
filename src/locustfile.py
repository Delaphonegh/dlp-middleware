from locust import HttpUser, task, between
import json

class ApiUser(HttpUser):
    wait_time = between(1, 3)
    token = None
    
    @task(3)
    def login(self):
        # Login using the provided credentials
        response = self.client.post("/auth/login", json={
            "username": "testuser1",
            "password": "password123",
            "company_code": "0FZXQO4Y"
        })
        
        # Store token for future requests
        if response.status_code == 200:
            result = response.json()
            if "access_token" in result:
                self.token = result["access_token"]
        else:
            print(f"Login failed with status {response.status_code}: {response.text}")
    
    @task(2)
    def chat(self):
        # Skip if no token
        if not self.token:
            return
            
        # Set authorization header
        headers = {"Authorization": f"Bearer {self.token}"}
        
        # Test chat endpoint
        response = self.client.post("/chat", 
            json={
                "question": "Analyze conversation data for the October 2024",
                "conversation_id": "4d3a93f7-36de-40ef-9db0-d17e7680e3e3"
            },
            headers=headers
        )
        
        if response.status_code != 200:
            print(f"Chat request failed with status {response.status_code}: {response.text}")