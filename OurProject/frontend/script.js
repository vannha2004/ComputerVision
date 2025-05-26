document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('imageUpload');
    const classifyBtn = document.getElementById('classifyBtn');
    const previewImage = document.getElementById('previewImage');
    const predictionText = document.getElementById('predictionText');
    
    // Preview image when uploaded
    imageUpload.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
                predictionText.textContent = 'Click Classify to predict';
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Classify button click handler
    classifyBtn.addEventListener('click', function() {
        const file = imageUpload.files[0];
        if (!file) {
            alert('Please upload an image first');
            return;
        }
        
        predictionText.textContent = 'Classifying...';
        
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            predictionText.textContent = `Prediction: ${data.prediction}`;
        })
        .catch(error => {
            console.error('Error:', error);
            predictionText.textContent = 'Error occurred during classification';
        });
    });
});