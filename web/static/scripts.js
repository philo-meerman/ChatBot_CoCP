// web/static/scripts.js
function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput }),
    })
    .then(response => response.json())
    .then(data => {
        const chatBox = document.getElementById('chat-box');
        const userMessage = document.createElement('div');
        userMessage.textContent = `User: ${userInput}`;
        const botResponse = document.createElement('div');
        botResponse.textContent = `Bot: ${data.response}`;
        chatBox.appendChild(userMessage);
        chatBox.appendChild(botResponse);
        document.getElementById('user-input').value = '';
    })
    .catch(error => console.error('Error:', error));
}
