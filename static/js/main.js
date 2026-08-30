// Main JavaScript for AI Resume Analyzer

document.addEventListener('DOMContentLoaded', function () {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('resume');
    const placeholder = document.getElementById('uploadPlaceholder');
    const preview = document.getElementById('uploadPreview');
    const fileName = document.getElementById('fileName');
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');

    if (!uploadZone || !fileInput) return;

    // File selection change
    fileInput.addEventListener('change', function () {
        handleFileSelect(this.files);
    });

    // Drag & Drop events
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, function (e) {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, function (e) {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.remove('dragover');
        });
    });

    uploadZone.addEventListener('drop', function (e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(files);
        }
    });

    function handleFileSelect(files) {
        if (files && files[0]) {
            const file = files[0];
            const validExtensions = ['pdf', 'docx'];
            const fileExt = file.name.split('.').pop().toLowerCase();

            if (!validExtensions.includes(fileExt)) {
                alert('Please upload a PDF or DOCX file.');
                fileInput.value = '';
                return;
            }

            if (file.size > 16 * 1024 * 1024) {
                alert('File size exceeds 16MB limit.');
                fileInput.value = '';
                return;
            }

            placeholder.classList.add('d-none');
            preview.classList.remove('d-none');
            fileName.textContent = file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
        }
    }

    // Show loading state on form submit
    if (uploadForm && analyzeBtn) {
        uploadForm.addEventListener('submit', function () {
            if (fileInput.files.length > 0) {
                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing Resume...';
            }
        });
    }
});
