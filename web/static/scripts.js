// web/static/scripts.js
function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === "") return;

    // Append the user's message to the chat box immediately
    const chatBox = document.getElementById('chat-box');
    const userMessage = document.createElement('div');
    userMessage.className = 'message user-message';
    userMessage.textContent = userInput;
    chatBox.appendChild(userMessage);

    // Clear the input box
    document.getElementById('user-input').value = '';

    // Scroll to the bottom of the chat box
    chatBox.scrollTop = chatBox.scrollHeight;

    // Show the loader where the bot response will appear
    const loader = document.createElement('div');
    loader.className = 'spinner';
    loader.id = 'loader';
    chatBox.appendChild(loader);

    // Fetch the response from the server
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput }),
    })
    .then(response => response.json())
    .then(data => {
        // Remove the loader
        chatBox.removeChild(loader);

        // Append the bot's response to the chat box
        const botResponse = document.createElement('div');
        botResponse.className = 'message bot-message';
        botResponse.innerHTML = data.response.replace(/\n/g, '<br>');
        chatBox.appendChild(botResponse);

        // Scroll to the bottom of the chat box
        chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
        // Remove the loader in case of an error
        chatBox.removeChild(loader);
    });
}

// Add event listener for Enter key to submit the message
document.getElementById('user-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();  // Prevent the default form submission
        sendMessage();
    }
});
