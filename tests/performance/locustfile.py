from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def test_chat(self):
        self.client.post("/chat", json={
            "messages": [{
                "type": "text",
                "content": "Hello"
            }]
        })
    
    @task(2)
    def test_translation(self):
        self.client.post("/translate", json={
            "text": "Hello",
            "target_lang": "fr"
        }) 