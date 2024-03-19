// index.js

document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('.login form');

    form.addEventListener('submit', function (event) {
        event.preventDefault();

        // Get the value from the input field
        const symbolInput = document.querySelector('.login form .form-control');
        const symbol = symbolInput.value.trim();

        // Make sure the symbol is not empty
        if (symbol !== '') {
            // Send a fetch request to the server
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol: symbol,
                }),
            })
            .then(response => response.json())
            .then(data => {
                // Handle the response data
                console.log(data);

                // You can update the UI or perform other actions based on the response
                // For example, show a message to the user or redirect to another page
            })
            .catch(error => {
                console.error('Error:', error);
                // Handle the error, show an error message, or perform other actions
            });
        } else {
            // Handle the case when the symbol is empty
            console.error('Symbol is empty. Please enter a valid symbol.');
            // Show an error message to the user or perform other actions
        }
    });
});
