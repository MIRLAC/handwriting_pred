// Get references to all the HTML elements we need to interact with
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const loader = document.getElementById('loader');
const errorMessage = document.getElementById('errorMessage');
const resultsContainer = document.getElementById('resultsContainer');

// Define the URL of your backend endpoint
const API_URL = 'http://127.0.0.1:8000/analyze';

// Add a click event listener to the "Analyze" button
analyzeBtn.addEventListener('click', async () => {
    // 1. Get the selected file
    const file = fileInput.files[0];
    if (!file) {
        errorMessage.textContent = 'Please select an image file first.';
        return;
    }

    // 2. Prepare for the API call: show loader, clear old results
    loader.classList.remove('hidden');
    errorMessage.textContent = '';
    resultsContainer.innerHTML = '';
    analyzeBtn.disabled = true;

    // 3. Create a FormData object to send the file
    const formData = new FormData();
    formData.append('file', file);

    try {
        // 4. Make the API call using fetch
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData,
        });

        // 5. Handle the response
        if (!response.ok) {
            // If the server returns an error (like 400 or 503), handle it
            const errorData = await response.json();
            throw new Error(errorData.detail || 'An unknown error occurred.');
        }

        // If the call was successful, get the JSON data
        const data = await response.json();
        displayResults(data);

    } catch (error) {
        // Display any errors that occurred during the fetch
        errorMessage.textContent = `Error: ${error.message}`;
    } finally {
        // 6. Clean up: hide the loader and re-enable the button
        loader.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
});

// A helper function to display the results in a nice format
function displayResults(data) {
    // Clear any previous results
    resultsContainer.innerHTML = '';
    
    // Create a title for the results section
    const title = document.createElement('h3');
    title.textContent = 'Analysis Results';
    resultsContainer.appendChild(title);

    // Loop through the main trait results and display them
    for (const key in data) {
        if (key !== 'diagnostics') {
            const item = document.createElement('div');
            item.className = 'result-item';

            const label = document.createElement('span');
            label.className = 'result-label';
            // Format the key to be more readable (e.g., "loops_per_100_chars" -> "Loops Per 100 Chars")
            label.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

            const value = document.createElement('span');
            value.className = 'result-value';
            // Format numbers to have 2 decimal places
            value.textContent = typeof data[key] === 'number' ? data[key].toFixed(2) : data[key];

            item.appendChild(label);
            item.appendChild(value);
            resultsContainer.appendChild(item);
        }
    }
    
    // Display the diagnostics section separately
    if (data.diagnostics) {
        const diagTitle = document.createElement('h4');
        diagTitle.textContent = 'Diagnostics';
        diagTitle.style.marginTop = '20px';
        resultsContainer.appendChild(diagTitle);

        for (const key in data.diagnostics) {
            const item = document.createElement('div');
            item.className = 'result-item';
            
            const label = document.createElement('span');
            label.className = 'result-label';
            label.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            
            const value = document.createElement('span');
            value.className = 'result-value';
            value.textContent = data.diagnostics[key];
            
            item.appendChild(label);
            item.appendChild(value);
            resultsContainer.appendChild(item);
        }
    }
}