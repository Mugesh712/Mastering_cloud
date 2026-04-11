/* ═══════════════════════════════════════════════════════════════
   PneumoCloud AI — Frontend JavaScript
   ═══════════════════════════════════════════════════════════════ */

let selectedFile = null;
let currentMode = 'demo';
let probChart = null;

// ── Page Navigation ──────────────────────────────────────────
function showPage(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));

    document.getElementById(`page-${page}`).classList.add('active');
    document.getElementById(`nav-${page}`).classList.add('active');

    if (page === 'records') loadRecords();
}

// ── Mode Toggle ──────────────────────────────────────────────
function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`mode-${mode}`).classList.add('active');
}

// ── File Upload ──────────────────────────────────────────────
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');

// Click to upload
uploadArea.addEventListener('click', (e) => {
    if (e.target.closest('.btn-secondary')) return;
    if (!selectedFile) fileInput.click();
});

// Drag & drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

// File selected
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

function handleFile(file) {
    const valid = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!valid.includes(file.type)) {
        alert('Please upload a JPEG or PNG image.');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('preview-image').src = e.target.result;
        document.getElementById('file-name').textContent = file.name;
        document.getElementById('upload-content').style.display = 'none';
        document.getElementById('upload-preview').style.display = 'block';
        document.getElementById('analyze-btn').disabled = false;
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    document.getElementById('upload-content').style.display = 'flex';
    document.getElementById('upload-preview').style.display = 'none';
    document.getElementById('analyze-btn').disabled = true;
    document.getElementById('results-section').style.display = 'none';

    // Reset pipeline
    document.querySelectorAll('.step-dot').forEach(d => {
        d.classList.remove('active', 'complete');
    });
    document.querySelector('#step-1 .step-dot').classList.add('active');
    document.querySelectorAll('.pipeline-line').forEach(l => l.classList.remove('active'));
}

// ── Analyse ──────────────────────────────────────────────────
async function analyzeImage() {
    if (!selectedFile) return;

    const btn = document.getElementById('analyze-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');
    const overlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');

    // UI: show loading
    btnText.style.display = 'none';
    btnLoading.style.display = 'flex';
    btn.disabled = true;
    overlay.style.display = 'flex';
    resultsSection.style.display = 'none';

    // Animate pipeline steps
    animatePipeline();

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('mode', currentMode);
        formData.append('patient_id', document.getElementById('patient-id').value);

        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.error) {
            alert(`Error: ${result.error}`);
            return;
        }

        // Complete pipeline animation
        completePipeline();

        // Show results after brief delay
        setTimeout(() => {
            overlay.style.display = 'none';
            displayResults(result);
        }, 800);

    } catch (err) {
        alert(`Error: ${err.message}`);
        overlay.style.display = 'none';
    } finally {
        btnText.style.display = 'flex';
        btnLoading.style.display = 'none';
        btn.disabled = false;
    }
}

// ── Pipeline Animation ───────────────────────────────────────
function animatePipeline() {
    const dots = document.querySelectorAll('.step-dot');
    const lines = document.querySelectorAll('.pipeline-line');
    const loadingText = document.getElementById('loading-text');

    const steps = [
        { delay: 0, text: 'Uploading X-ray image...' },
        { delay: 500, text: 'AWS Lambda processing...' },
        { delay: 1200, text: 'Running DenseNet-121 inference...' },
        { delay: 2000, text: 'Storing record in Azure...' },
        { delay: 2800, text: 'Preparing results...' },
    ];

    // Show scan effect
    showScanEffect();

    steps.forEach((step, i) => {
        setTimeout(() => {
            if (i > 0) {
                dots[i - 1].classList.remove('active');
                dots[i - 1].classList.add('complete');
                if (lines[i - 1]) lines[i - 1].classList.add('active');
            }
            dots[i].classList.add('active');
            loadingText.textContent = step.text;
        }, step.delay);
    });
}

function completePipeline() {
    document.querySelectorAll('.step-dot').forEach(d => {
        d.classList.remove('active');
        d.classList.add('complete');
    });
    document.querySelectorAll('.pipeline-line').forEach(l => l.classList.add('active'));
}

// ── Display Results ──────────────────────────────────────────
function displayResults(result) {
    hideScanEffect();

    const section = document.getElementById('results-section');
    section.style.display = 'block';
    setTimeout(() => {
        document.getElementById('result-header').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    const disease = result.diagnosis || 'Unknown';
    const confidence = result.confidence || 0;
    const riskScore = result.risk_score || 0;
    const triageLevel = result.triage_level || 'UNKNOWN';
    const department = result.department || 'N/A';
    const isNormal = disease === 'Normal';

    // ── Severity mapping ──
    const severityMap = {
        'COVID': 'critical',
        'Lung_Opacity': 'moderate',
        'Normal': 'none',
        'Viral Pneumonia': 'high',
    };
    const severity = severityMap[disease] || 'moderate';

    // ── Result Header ──
    const header = document.getElementById('result-header');
    header.className = `result-header severity-${severity}`;

    const icons = {
        none: '✅', low: '🟢', moderate: '🟡', high: '🟠', critical: '🔴'
    };

    document.getElementById('diagnosis-icon').textContent = icons[severity] || '⚪';
    document.getElementById('diagnosis-name').textContent = disease;

    const colorMap = {
        'Normal': '#059669',
        'COVID': '#dc2626',
        'Viral Pneumonia': '#ea580c',
        'Lung_Opacity': '#d97706',
    };
    document.getElementById('diagnosis-name').style.color = colorMap[disease] || '#2563eb';

    document.getElementById('diagnosis-confidence').textContent =
        `Confidence: ${(confidence * 100).toFixed(1)}%`;
    document.getElementById('model-info').textContent = result.ai_model || 'AI Model';

    // ── Risk Score ──
    document.getElementById('risk-value').textContent = riskScore.toFixed(3);
    const riskBar = document.getElementById('risk-bar');
    riskBar.style.width = `${riskScore * 100}%`;
    riskBar.style.background =
        riskScore >= 0.8 ? 'linear-gradient(90deg, #dc2626, #b91c1c)' :
        riskScore >= 0.5 ? 'linear-gradient(90deg, #ea580c, #c2410c)' :
        riskScore >= 0.25 ? 'linear-gradient(90deg, #d97706, #b45309)' :
        'linear-gradient(90deg, #059669, #047857)';

    const riskLevel = document.getElementById('risk-level');
    riskLevel.textContent =
        riskScore >= 0.8 ? '🔴 HIGH RISK' :
        riskScore >= 0.5 ? '🟠 MODERATE RISK' :
        riskScore >= 0.25 ? '🟡 LOW-MODERATE' :
        '🟢 LOW RISK';
    riskLevel.style.color =
        riskScore >= 0.8 ? '#dc2626' :
        riskScore >= 0.5 ? '#ea580c' :
        riskScore >= 0.25 ? '#d97706' :
        '#059669';

    // ── Triage Badge ──
    const triageBadge = document.getElementById('triage-badge');
    triageBadge.textContent = triageLevel;
    triageBadge.className = `triage-badge triage-${triageLevel}`;
    document.getElementById('triage-dept').textContent = `📍 ${department}`;

    // ── Azure Status ──
    const azureEl = document.getElementById('azure-status');
    const azureStatus = result.azure_status || 'N/A';
    azureEl.textContent = azureStatus.includes('SAVED') ? '✅ SAVED' :
                          azureStatus.includes('DEMO') ? '🎮 DEMO' : '❌ ' + azureStatus;
    azureEl.style.background = azureStatus.includes('SAVED') ?
        '#f0fdf4' : '#eff6ff';
    azureEl.style.color = azureStatus.includes('SAVED') ? '#059669' : '#2563eb';

    // ── Probability Chart ──
    renderChart(result.all_probabilities || {});

    // ── Report Metadata ──
    document.getElementById('report-patient').textContent = `Patient: ${result.patient_id || 'N/A'}`;
    document.getElementById('report-date').textContent = `Date: ${new Date().toLocaleDateString('en-IN', {day:'2-digit', month:'short', year:'numeric'})}`;
    document.getElementById('report-model').textContent = `Model: ${result.ai_model || 'AI Model'}`;

    // ── Clinical Report ──
    const summaryEl = document.getElementById('summary-content');
    let summary = result.ai_summary || 'No clinical report available.';

    const processedHtml = processReportSummary(summary);
    summaryEl.innerHTML = processedHtml;

    // Store result globally for download
    window._lastResult = result;
    window._lastSummary = summary;

    // Show email section
    document.getElementById('email-section').style.display = 'block';
    document.getElementById('email-status').textContent = '';
    document.getElementById('email-status').className = 'email-status';
}

// ── Section emoji mapping ────────────────────────────────────
const SECTION_EMOJIS = {
    'DIAGNOSTIC FINDINGS':        { icon: '🔬', bullets: ['🩺', '📊', '💡', '🧬', '🔬'] },
    'CHEST X-RAY ANALYSIS':       { icon: '🔍', bullets: ['🔍', '📸', '🫁', '❤️', '👁️'] },
    'SEVERITY ASSESSMENT':        { icon: '⚠️', bullets: ['⚠️', '🏥', '🏢', '📌', '🚑'] },
    'LIFESTYLE RECOMMENDATIONS':  { icon: '🏃', bullets: ['🏃', '😴', '🏠', '🧘', '🌿'] },
    'DIETARY GUIDELINES':         { icon: '🍎', bullets: ['🍎', '🥗', '💧', '🚫', '💊'] },
    'FOLLOW-UP SCHEDULE':         { icon: '📅', bullets: ['📅', '🔄', '🧪', '🚨', '📋'] },
    'PROGNOSIS':                  { icon: '✅', bullets: ['✅', '⚡', '🌟', '💪', '🎯'] },
};

function getSectionKey(title) {
    const upper = title.toUpperCase();
    for (const key of Object.keys(SECTION_EMOJIS)) {
        if (upper.includes(key)) return key;
    }
    return null;
}

// ── Process Report Summary ───────────────────────────────────
function processReportSummary(rawText) {
    let text = rawText;
    text = text.replace(/(?:##\s*)?\d+\.\s*TREATMENT\s+PLAN\s*[&]\s*CLINICAL\s+ACTIONS[\s\S]*?(?=(?:##\s*)?\d+\.\s+[A-Z]|$)/gi, '');
    text = text.trim();

    const sectionRegex = /(?:##\s*)?(\d+)\.\s+([A-Z][A-Z\s&]+)/g;
    const sections = [];
    let match;
    const headerPositions = [];

    while ((match = sectionRegex.exec(text)) !== null) {
        headerPositions.push({
            index: match.index,
            fullMatch: match[0],
            num: match[1],
            title: match[2].trim(),
        });
    }

    for (let i = 0; i < headerPositions.length; i++) {
        const start = headerPositions[i].index + headerPositions[i].fullMatch.length;
        const end = (i + 1 < headerPositions.length) ? headerPositions[i + 1].index : text.length;
        const content = text.substring(start, end).trim();
        sections.push({
            title: headerPositions[i].title,
            content: content,
        });
    }

    let html = '<div class="report-body">';

    sections.forEach((section, idx) => {
        const sectionNum = idx + 1;
        const sectionKey = getSectionKey(section.title);
        const emojiSet = sectionKey ? SECTION_EMOJIS[sectionKey] : { icon: '📄', bullets: ['•', '•', '•', '•', '•'] };

        html += `<div class="report-section-header"><span class="section-num">${sectionNum}</span> ${section.title}</div>`;

        const bullets = convertToPoints(section.content);

        bullets.forEach((point, pIdx) => {
            const emoji = emojiSet.bullets[pIdx % emojiSet.bullets.length];
            html += `<div class="report-bullet">${emoji} ${point}</div>`;
        });
    });

    html += '</div>';
    return html;
}

// ── Convert paragraph/text content to bullet points ──────────
function convertToPoints(content) {
    if (!content) return [];

    const lines = content.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    const bulletLines = lines.filter(l => /^[-•🔬🩺📊💡🔍⚠️🏥🏢📌🏃😴🏠🧘🍎🥗💧🚫💊📅🔄🧪🚨📋✅⚡🌟💪]/.test(l));

    if (bulletLines.length >= lines.length * 0.6 && bulletLines.length >= 2) {
        return bulletLines.map(l => l.replace(/^[-•]\s*/, '').replace(/^[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]\s*/u, '').trim()).filter(l => l.length > 0);
    }

    const fullText = lines.join(' ');
    const sentences = fullText
        .split(/(?<=\.)\s+(?=[A-Z])/)
        .map(s => s.trim())
        .filter(s => s.length > 15);

    if (sentences.length <= 1 && fullText.length > 100) {
        const commaSplit = fullText.split(/,\s+/).map(s => s.trim()).filter(s => s.length > 15);
        if (commaSplit.length > 2) return commaSplit;
    }

    return sentences.length > 0 ? sentences : [fullText];
}

// ── Chart ─────────────────────────────────────────────────────
function renderChart(probs) {
    if (probChart) probChart.destroy();

    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    const labels = sorted.map(([k]) => k);
    const values = sorted.map(([, v]) => v * 100);

    const barColors = labels.map(label => {
        const severityColors = {
            'COVID': '#dc2626',
            'Viral Pneumonia': '#ea580c',
            'Lung_Opacity': '#d97706',
            'Normal': '#059669',
        };
        return severityColors[label] || '#2563eb';
    });

    const ctx = document.getElementById('prob-chart').getContext('2d');
    probChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: barColors.map(c => c + '22'),
                borderColor: barColors,
                borderWidth: 2,
                borderRadius: 6,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.x.toFixed(2)}%`
                    },
                    backgroundColor: '#1e293b',
                    titleColor: '#f8fafc',
                    bodyColor: '#94a3b8',
                    borderColor: '#334155',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 10,
                }
            },
            scales: {
                x: {
                    max: 100,
                    grid: { color: '#f1f5f9' },
                    ticks: { color: '#94a3b8', callback: v => v + '%', font: { size: 11 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#475569', font: { size: 12, weight: 500 } }
                }
            }
        }
    });
}

// ── Records Page ─────────────────────────────────────────────
async function loadRecords() {
    const content = document.getElementById('records-content');
    content.innerHTML = '<div class="loading-records"><span class="spinner"></span> Fetching records from Azure...</div>';

    try {
        const response = await fetch('/api/records');
        const data = await response.json();
        const records = data.records || [];

        if (records.length === 0) {
            content.innerHTML = `
                <div style="text-align:center; padding:3rem; color:var(--text-secondary)">
                    <p style="font-size:48px; margin-bottom:16px">📭</p>
                    <p>No records found. Run an analysis in Cloud Pipeline mode first.</p>
                    ${data.error ? `<p style="color:var(--text-muted); margin-top:8px; font-size:13px">Error: ${data.error}</p>` : ''}
                </div>`;
            return;
        }

        const total = records.length;
        const abnormal = records.filter(r => r.diagnosis !== 'Normal').length;
        const normal = total - abnormal;

        const diseaseCounts = {};
        records.forEach(r => {
            const d = r.diagnosis || 'Unknown';
            diseaseCounts[d] = (diseaseCounts[d] || 0) + 1;
        });
        const topDisease = Object.entries(diseaseCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'N/A';

        let html = `
            <div class="records-stats">
                <div class="stat-card"><div class="stat-value">${total}</div><div class="stat-label">Total Records</div></div>
                <div class="stat-card"><div class="stat-value" style="color:#dc2626">${abnormal}</div><div class="stat-label">Abnormal</div></div>
                <div class="stat-card"><div class="stat-value" style="color:#059669">${normal}</div><div class="stat-label">Normal</div></div>
                <div class="stat-card"><div class="stat-value" style="color:#d97706; font-size:18px">${topDisease}</div><div class="stat-label">Top Disease</div></div>
            </div>
        `;

        records.forEach(rec => {
            const diag = rec.diagnosis || 'N/A';
            const triage = rec.triage_level || 'N/A';
            const isNormal = diag === 'Normal';
            const triageClass = `triage-${triage}`;

            let time = rec.timestamp || 'N/A';
            if (time.length > 19) time = time.substring(0, 19).replace('T', ' ');

            html += `
                <div class="record-card" id="record-${rec.record_id}">
                    <div class="record-id">${rec.record_id || 'N/A'}</div>
                    <div class="record-patient">${rec.patient_id || 'N/A'}</div>
                    <div class="record-diagnosis" style="color:${isNormal ? '#059669' : '#dc2626'}">${isNormal ? '🟢' : '🔴'} ${diag}</div>
                    <div><span class="record-triage ${triageClass}">${triage}</span></div>
                    <div class="record-time">🕐 ${time}</div>
                    <button class="btn-delete-record" onclick="deleteRecord('${rec.record_id}')" title="Delete this record">
                        🗑️
                    </button>
                </div>
            `;
        });

        content.innerHTML = html;

    } catch (err) {
        content.innerHTML = `<div style="text-align:center; padding:3rem; color:#dc2626">❌ Error: ${err.message}</div>`;
    }
}

// ── Delete Record ────────────────────────────────────────────
async function deleteRecord(recordId) {
    if (!confirm(`Are you sure you want to delete record ${recordId}?\n\nThis will permanently remove it from Azure Cosmos DB.`)) {
        return;
    }

    const cardEl = document.getElementById(`record-${recordId}`);
    if (cardEl) {
        cardEl.style.opacity = '0.4';
        cardEl.style.pointerEvents = 'none';
    }

    try {
        const response = await fetch('/api/records/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ record_id: recordId })
        });

        const data = await response.json();

        if (data.success) {
            if (cardEl) {
                cardEl.style.transition = 'all 0.4s ease';
                cardEl.style.transform = 'translateX(100%)';
                cardEl.style.opacity = '0';
                cardEl.style.maxHeight = '0';
                cardEl.style.padding = '0';
                cardEl.style.margin = '0';
                setTimeout(() => {
                    cardEl.remove();
                    loadRecords();
                }, 500);
            }
        } else {
            alert(`❌ Failed to delete: ${data.error || 'Unknown error'}`);
            if (cardEl) {
                cardEl.style.opacity = '1';
                cardEl.style.pointerEvents = 'auto';
            }
        }
    } catch (err) {
        alert(`❌ Error: ${err.message}`);
        if (cardEl) {
            cardEl.style.opacity = '1';
            cardEl.style.pointerEvents = 'auto';
        }
    }
}

// ── Download Report as PDF ───────────────────────────────────
function downloadReport() {
    if (!window._lastResult) {
        alert('No report to download. Run an analysis first.');
        return;
    }

    const r = window._lastResult;
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF('p', 'mm', 'a4');
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 18;
    const contentWidth = pageWidth - margin * 2;
    let y = 20;

    function stripEmojis(text) {
        return text.replace(/[\u{1F600}-\u{1F9FF}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{FE00}-\u{FE0F}\u{200D}\u{20E3}\u{E0020}-\u{E007F}\u{1F1E0}-\u{1F1FF}]/gu, '').replace(/^\s+/, '');
    }

    function checkPage(needed) {
        if (y + needed > pageHeight - 20) {
            doc.addPage();
            y = 20;
        }
    }

    function drawRect(x, yPos, w, h, color) {
        doc.setFillColor(color[0], color[1], color[2]);
        doc.rect(x, yPos, w, h, 'F');
    }

    // ── HEADER BANNER ──
    drawRect(0, 0, pageWidth, 42, [37, 99, 235]);
    drawRect(0, 42, pageWidth, 2, [29, 78, 216]);

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(22);
    doc.setFont('helvetica', 'bold');
    doc.text('PneumoCloud AI', margin, 18);

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text('Multi-Cloud Chest X-Ray Classification System', margin, 26);

    doc.setFontSize(9);
    doc.text('CLINICAL ANALYSIS REPORT', margin, 34);

    const dateStr = new Date().toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
    doc.setFontSize(9);
    doc.text(dateStr, pageWidth - margin, 34, { align: 'right' });

    y = 52;

    // ── PATIENT INFO BOX ──
    drawRect(margin, y, contentWidth, 28, [241, 245, 249]);
    doc.setDrawColor(37, 99, 235);
    doc.setLineWidth(0.5);
    doc.rect(margin, y, contentWidth, 28, 'S');

    doc.setTextColor(30, 41, 59);
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.text('PATIENT DETAILS', margin + 5, y + 7);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9);
    doc.setTextColor(71, 85, 105);

    const patientInfo = [
        `Patient ID: ${r.patient_id || 'N/A'}`,
        `AI Model: ${r.ai_model || 'DenseNet-121'}`,
        `Date: ${dateStr}`,
    ];
    const col2Info = [
        `Diagnosis: ${r.diagnosis || 'N/A'}`,
        `Confidence: ${((r.confidence || 0) * 100).toFixed(1)}%`,
        `Mode: Cloud Pipeline`,
    ];

    patientInfo.forEach((line, i) => {
        doc.text(line, margin + 5, y + 14 + i * 5);
    });
    col2Info.forEach((line, i) => {
        doc.text(line, margin + contentWidth / 2, y + 14 + i * 5);
    });

    y += 35;

    // ── DIAGNOSIS SUMMARY BOX ──
    const isNormal = r.diagnosis === 'Normal';
    const diagColor = isNormal ? [5, 150, 105] : [220, 38, 38];
    const diagBg = isNormal ? [240, 253, 244] : [254, 242, 242];

    drawRect(margin, y, contentWidth, 22, diagBg);
    doc.setDrawColor(diagColor[0], diagColor[1], diagColor[2]);
    doc.setLineWidth(1);
    doc.line(margin, y, margin, y + 22);

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(14);
    doc.setTextColor(diagColor[0], diagColor[1], diagColor[2]);
    doc.text(r.diagnosis || 'N/A', margin + 6, y + 10);

    doc.setFontSize(9);
    doc.setTextColor(80, 80, 80);
    doc.setFont('helvetica', 'normal');
    doc.text(`Risk: ${(r.risk_score || 0).toFixed(3)}  |  Triage: ${r.triage_level || 'N/A'}  |  Dept: ${r.department || 'N/A'}`, margin + 6, y + 18);

    y += 30;

    // ── PROBABILITY TABLE ──
    if (r.all_probabilities) {
        checkPage(30);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(11);
        doc.setTextColor(37, 99, 235);
        doc.text('DISEASE PROBABILITY DISTRIBUTION', margin, y);
        y += 6;

        const sorted = Object.entries(r.all_probabilities).sort((a, b) => b[1] - a[1]);

        drawRect(margin, y, contentWidth, 7, [241, 245, 249]);
        doc.setFontSize(8);
        doc.setTextColor(71, 85, 105);
        doc.setFont('helvetica', 'bold');
        doc.text('Disease', margin + 3, y + 5);
        doc.text('Probability', margin + 60, y + 5);
        doc.text('Bar', margin + 95, y + 5);
        y += 8;

        doc.setFont('helvetica', 'normal');
        sorted.forEach(([name, prob]) => {
            doc.setTextColor(71, 85, 105);
            doc.text(name, margin + 3, y + 4);
            doc.text(`${(prob * 100).toFixed(1)}%`, margin + 60, y + 4);

            const barWidth = prob * 70;
            const barColorMap = {
                'COVID': [220, 38, 38], 'Viral Pneumonia': [234, 88, 12],
                'Lung_Opacity': [217, 119, 6], 'Normal': [5, 150, 105]
            };
            const bc = barColorMap[name] || [37, 99, 235];
            drawRect(margin + 95, y, barWidth, 5, bc);
            y += 7;
        });
        y += 6;
    }

    // ── CLINICAL REPORT SECTIONS ──
    const rawSummary = window._lastSummary || 'No detailed report available.';

    let cleanedText = rawSummary;
    cleanedText = cleanedText.replace(/(?:##\s*)?\d+\.\s*TREATMENT\s+PLAN\s*[&]\s*CLINICAL\s+ACTIONS[\s\S]*?(?=(?:##\s*)?\d+\.\s+[A-Z]|$)/gi, '').trim();

    const pdfSectionRegex = /(?:##\s*)?(\d+)\.\s+([A-Z][A-Z\s&]+)/g;
    const pdfHeaders = [];
    let pdfMatch;
    while ((pdfMatch = pdfSectionRegex.exec(cleanedText)) !== null) {
        pdfHeaders.push({ index: pdfMatch.index, fullMatch: pdfMatch[0], title: pdfMatch[2].trim() });
    }

    const pdfSectionColors = {
        'DIAGNOSTIC FINDINGS': [37, 99, 235],
        'CHEST X-RAY ANALYSIS': [124, 58, 237],
        'SEVERITY ASSESSMENT': [217, 119, 6],
        'LIFESTYLE RECOMMENDATIONS': [5, 150, 105],
        'DIETARY GUIDELINES': [219, 39, 119],
        'FOLLOW-UP SCHEDULE': [234, 88, 12],
        'PROGNOSIS': [6, 182, 212],
    };

    function getPdfSectionColor(title) {
        const upper = title.toUpperCase();
        for (const [key, color] of Object.entries(pdfSectionColors)) {
            if (upper.includes(key)) return color;
        }
        return [37, 99, 235];
    }

    pdfHeaders.forEach((hdr, idx) => {
        const start = hdr.index + hdr.fullMatch.length;
        const end = (idx + 1 < pdfHeaders.length) ? pdfHeaders[idx + 1].index : cleanedText.length;
        const sectionContent = cleanedText.substring(start, end).trim();
        const sectionNum = idx + 1;
        const color = getPdfSectionColor(hdr.title);

        checkPage(14);
        y += 3;
        drawRect(margin, y, contentWidth, 8, color);
        doc.setTextColor(255, 255, 255);
        doc.setFontSize(9);
        doc.setFont('helvetica', 'bold');
        doc.text(`${sectionNum}. ${stripEmojis(hdr.title).toUpperCase()}`, margin + 4, y + 5.5);
        y += 12;

        const bullets = convertToPoints(sectionContent);

        bullets.forEach(point => {
            const cleanText = stripEmojis(point.replace(/^[-•]\s*/, ''));
            if (!cleanText) return;

            checkPage(8);

            doc.setFillColor(color[0], color[1], color[2]);
            doc.circle(margin + 3, y + 1.5, 1, 'F');

            doc.setTextColor(50, 55, 70);
            doc.setFontSize(8.5);
            doc.setFont('helvetica', 'normal');
            const splitLines = doc.splitTextToSize(cleanText, contentWidth - 12);
            splitLines.forEach(sl => {
                checkPage(5);
                doc.text(sl, margin + 8, y + 2);
                y += 4.5;
            });
            y += 1;
        });
    });

    // ── FOOTER ──
    checkPage(25);
    y += 8;
    doc.setDrawColor(200, 200, 220);
    doc.setLineWidth(0.3);
    doc.line(margin, y, pageWidth - margin, y);
    y += 6;

    doc.setFontSize(7.5);
    doc.setTextColor(120, 130, 150);
    doc.setFont('helvetica', 'italic');
    doc.text('This report was generated by PneumoCloud AI using DenseNet-121 deep learning model.', margin, y);
    y += 4;
    doc.text('For clinical decisions, always consult a qualified healthcare professional.', margin, y);
    y += 4;
    doc.text('PneumoCloud AI  |  Multi-Cloud Pipeline: AWS + GCP + Azure', margin, y);

    const totalPages = doc.internal.getNumberOfPages();
    for (let i = 1; i <= totalPages; i++) {
        doc.setPage(i);
        doc.setFontSize(8);
        doc.setTextColor(150, 150, 170);
        doc.setFont('helvetica', 'normal');
        doc.text(`Page ${i} of ${totalPages}`, pageWidth - margin, pageHeight - 10, { align: 'right' });
        doc.text('PneumoCloud AI Report', margin, pageHeight - 10);
    }

    const filename = `PneumoCloud_Report_${r.patient_id || 'patient'}_${new Date().toISOString().slice(0, 10)}.pdf`;
    doc.save(filename);
}

// ── Send Report via Email ────────────────────────────────────
async function sendReportEmail() {
    const emailInput = document.getElementById('email-input');
    const statusEl = document.getElementById('email-status');
    const btn = document.getElementById('send-email-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');
    const email = emailInput.value.trim();

    if (!email) {
        statusEl.textContent = '⚠️ Please enter an email address.';
        statusEl.className = 'email-status error';
        emailInput.focus();
        return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        statusEl.textContent = '⚠️ Please enter a valid email address.';
        statusEl.className = 'email-status error';
        emailInput.focus();
        return;
    }

    if (!window._lastResult) {
        statusEl.textContent = '⚠️ No report available. Run an analysis first.';
        statusEl.className = 'email-status error';
        return;
    }

    btnText.style.display = 'none';
    btnLoading.style.display = 'flex';
    btn.disabled = true;
    statusEl.textContent = '📤 Sending report...';
    statusEl.className = 'email-status sending';

    try {
        const response = await fetch('/api/send-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                email: email,
                result: window._lastResult,
                summary: window._lastSummary,
            })
        });

        const data = await response.json();

        if (data.success) {
            statusEl.textContent = `✅ Report sent successfully to ${email}!`;
            statusEl.className = 'email-status success';
        } else {
            statusEl.textContent = `❌ Failed to send: ${data.error || 'Unknown error'}`;
            statusEl.className = 'email-status error';
        }
    } catch (err) {
        statusEl.textContent = `❌ Error: ${err.message}`;
        statusEl.className = 'email-status error';
    } finally {
        btnText.style.display = 'flex';
        btnLoading.style.display = 'none';
        btn.disabled = false;
    }
}

// ── Scanning state helpers ───────────────────────────────────
function showScanEffect() {
    const preview = document.getElementById('upload-preview');
    const scanText = document.getElementById('scan-text');
    if (preview) preview.classList.add('scanning');
    if (scanText) scanText.classList.add('active');
}

function hideScanEffect() {
    const preview = document.getElementById('upload-preview');
    const scanText = document.getElementById('scan-text');
    if (preview) preview.classList.remove('scanning');
    if (scanText) scanText.classList.remove('active');
}

// ── Initialize ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    document.querySelector('#step-1 .step-dot').classList.add('active');

    // Stagger fade-up animations
    const staggerItems = document.querySelectorAll('.stagger-item');
    staggerItems.forEach((item, i) => {
        item.style.transitionDelay = `${i * 0.08}s`;
        setTimeout(() => {
            item.classList.add('visible');
        }, 50);
    });

    // Scroll observer for elements
    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });

        staggerItems.forEach(item => observer.observe(item));
    }
});
