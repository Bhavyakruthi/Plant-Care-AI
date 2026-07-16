import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, Upload, Zap, Image as ImageIcon, FileText, Moon, Sun, Leaf, Grid, BookOpen, ChevronRight, BarChart } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [mode, setMode] = useState('landing'); // 'landing', 'analysis', 'gallery'
  const [darkMode, setDarkMode] = useState(true);
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [galAssets, setGalAssets] = useState({ visualizations: [], reports: [] });
  const [galleryFilter, setGalleryFilter] = useState('all');
  const [selectedReport, setSelectedReport] = useState(null);
  const [reportContent, setReportContent] = useState('');
  const [reportLoading, setReportLoading] = useState(false);

  useEffect(() => {
    fetchGallery();
  }, []);

  const fetchGallery = async () => {
    try {
      const res = await axios.get(`${API_BASE}/results`);
      setGalAssets(res.data);
    } catch (err) {
      console.error("Failed to load gallery data", err);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleDiagnose = async () => {
    if (!selectedFile) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE}/diagnose`, formData);
      setResult(response.data);
    } catch (error) {
      console.error('Diagnosis failed:', error);
      alert('Backend connection failed. Please ensure FastAPI is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const handleViewReport = async (rep) => {
    console.log("Report object:", rep);
    console.log("Constructed URL:", `${API_BASE}${rep.url}`);
    setSelectedReport(rep);
    setReportLoading(true);
    setReportContent('');
    try {
      const response = await axios.get(`${API_BASE}${rep.url}`);
      console.log("Response received:", response.status);
      setReportContent(response.data);
    } catch (err) {
      console.error("Failed to load report", err);
      console.error("Error details:", err.response?.status, err.response?.data);
      setReportContent("Failed to load scientific report. Please ensure the backend is running.");
    } finally {
      setReportLoading(false);
    }
  };

  const categories = ['all', ...new Set(galAssets.visualizations.map(v => v.category))];

  // Components
  const Nav = () => (
    <header className="analysis-header">
      <div className="logo-section" onClick={() => setMode('landing')} style={{ cursor: 'pointer' }}>
        <div className="logo-icon-small">
          <Leaf size={24} />
        </div>
        <div>
          <h1 className="brand-name-small">PlantCare AI</h1>
          <p className="brand-tagline-small">Scientific Diagnosis</p>
        </div>
      </div>
      <nav className="header-nav">
        <button className={`nav-btn ${mode === 'analysis' ? 'active' : ''}`} onClick={() => setMode('analysis')}>
          <Zap size={16} /> Diagnosis
        </button>
        <button className={`nav-btn ${mode === 'gallery' ? 'active' : ''}`} onClick={() => setMode('gallery')}>
          <Grid size={16} /> Gallery
        </button>
      </nav>
      <div className="header-actions">
        <button className="theme-toggle" onClick={() => setDarkMode(!darkMode)}>
          {darkMode ? <Sun size={20} /> : <Moon size={20} />}
        </button>
        <div className="login-btn">97.2% Acc</div>
      </div>
    </header>
  );

  const LandingPage = () => (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className={`landing-page ${darkMode ? 'dark' : 'light'}`}
    >
      <header className="landing-header">
        <div className="logo-section">
          <div className="logo-icon">
            <Leaf size={32} />
          </div>
          <div>
            <h1 className="brand-name">PlantCare AI</h1>
            <p className="brand-tagline">Quantum-Enhanced Fusion</p>
          </div>
        </div>
        <div className="header-actions">
          <button className="theme-toggle" onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
          <button className="login-btn" onClick={() => setMode('gallery')}>View Portfolio</button>
        </div>
      </header>

      <main className="hero-section">
        <div className="badge">
          <BarChart size={16} />
          <span>97.24% Multimodal Accuracy Breakthrough</span>
        </div>
        <h2 className="hero-title">
          The Future of
          <br />
          <span className="highlight">Botanical Diagnosis</span>
        </h2>
        <p className="hero-subtitle">
          Overhauled BioBERT Hybrid System and Quantum-Classical CNN for
          unprecedented accuracy in plant disease identification.
        </p>

        <div className="feature-cards">
          <div className="feature-card" onClick={() => setMode('analysis')}>
            <div className="feature-icon green">
              <Camera size={24} />
            </div>
            <h3>Real-time Inference</h3>
            <p>Full-stack computer vision and NLP diagnostic pipeline.</p>
          </div>
          <div className="feature-card" onClick={() => setMode('gallery')}>
            <div className="feature-icon purple">
              <Grid size={24} />
            </div>
            <h3>Explainability Gallery</h3>
            <p>Grad-CAM, Attention Maps, and Feature Space visualizations.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon yellow">
              <BookOpen size={24} />
            </div>
            <h3>Scientific Reports</h3>
            <p>Complete 10-fold CV results and technical performance audits.</p>
          </div>
        </div>

        <button className="cta-button" onClick={() => setMode('analysis')}>
          Launch Diagnostic Engine
        </button>
      </main>
    </motion.div>
  );

  const GalleryPage = () => (
    <motion.div
      initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
      className={`analysis-page ${darkMode ? 'dark' : 'light'}`}
    >
      <Nav />
      <main className="analysis-content">
        <div className="section-header">
          <Grid size={24} color="#10b981" />
          <h2 className="page-title">Project Gallery & Reports</h2>
        </div>
        <p className="page-subtitle">Scientific evidence, interpretability visualizations, and final benchmarking results.</p>

        <div className="tab-container">
          {categories.map(cat => (
            <button key={cat} className={`tab ${galleryFilter === cat ? 'active' : ''}`} onClick={() => setGalleryFilter(cat)}>
              {cat.charAt(0).toUpperCase() + cat.slice(1).replace('_', ' ')}
            </button>
          ))}
        </div>

        <div className="gallery-grid">
          <AnimatePresence mode='popLayout'>
            {galAssets.visualizations
              .filter(v => galleryFilter === 'all' || v.category === galleryFilter)
              .map((v, i) => (
                <motion.div
                  key={v.url} layout initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: i * 0.05 }}
                  className="gallery-card"
                >
                  <div className="gallery-img-container">
                    <img src={`${API_BASE}${v.url}`} alt={v.name} className="gallery-img" />
                    <div className="gallery-overlay">
                      <ImageIcon color="white" size={24} />
                    </div>
                  </div>
                  <div className="gallery-info">
                    <span className="gallery-category">{v.category}</span>
                    <div className="gallery-name">{v.name}</div>
                  </div>
                </motion.div>
              ))}
          </AnimatePresence>
        </div>

        <div className="reports-section" style={{ marginTop: '60px' }}>
          <div className="section-header">
            <FileText size={24} color="#10b981" />
            <h3 style={{ margin: 0 }}>Scientific Performance Reports</h3>
          </div>
          <div className="reports-grid" style={{ marginTop: '20px' }}>
            {galAssets.reports.map(rep => (
              <button key={rep.url} className="report-item" onClick={() => handleViewReport(rep)}>
                <div className="report-icon"><FileText size={20} /></div>
                <div className="report-details">
                  <h4>{rep.name}</h4>
                  <span>Final Submission Document</span>
                </div>
                <ChevronRight style={{ marginLeft: 'auto' }} size={16} />
              </button>
            ))}
          </div>
        </div>
      </main>
    </motion.div>
  );

  const AnalysisPage = () => (
    <motion.div
      initial={{ x: 20, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
      className={`analysis-page ${darkMode ? 'dark' : 'light'}`}
    >
      <Nav />
      <main className="analysis-content">
        <h2 className="page-title">Live Diagnostic Engine</h2>
        <p className="page-subtitle">Utilizing Confidence-Aware Dynamic Fusion for high-precision botanical analysis.</p>

        <div className="analysis-container">
          <div className="input-section">
            <div className="step-section">
              <div className="upload-options">
                <button className="upload-card" onClick={() => document.getElementById('fileInput').click()}>
                  <Camera size={32} />
                  <span>Start Camera</span>
                </button>
                <button className="upload-card" onClick={() => document.getElementById('fileInput').click()}>
                  <Upload size={32} />
                  <span>Browse Files</span>
                </button>
              </div>
              <input type="file" id="fileInput" hidden onChange={handleFileChange} accept="image/*" />
              {preview && (
                <div className="preview-container">
                  <img src={preview} alt="Preview" className="preview-image" />
                </div>
              )}
            </div>

            <button className="analyze-button" onClick={handleDiagnose} disabled={!selectedFile || loading}>
              <Zap size={20} />
              {loading ? 'Processing Pipeline...' : 'Run Diagnostics'}
            </button>
          </div>

          <div className="results-panel">
            {result ? (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="results-wrapper">
                <div className="result-header">
                  <span className="result-tag">Multimodal Diagnosis</span>
                  <div className="disease-label">{result.disease}</div>
                </div>

                <div className="confidence-section">
                  <div className="confidence-meta">
                    <span className="confidence-label">System Confidence</span>
                    <span className="confidence-value">{result.confidence}</span>
                  </div>
                  <div className="confidence-bar">
                    <motion.div
                      initial={{ width: 0 }} animate={{ width: result.confidence }}
                      className="confidence-fill"
                    />
                  </div>
                </div>

                <div className="xai-grid">
                  <div className="xai-card">
                    <h4>Visual Focus (Grad-CAM)</h4>
                    <img src={`data:image/png;base64,${result.heatmap}`} alt="Heatmap" className="heatmap" />
                  </div>
                  <div className="xai-card">
                    <h4>Symptom Distribution</h4>
                    <div className="importance-list">
                      {Object.entries(result.importance || {}).map(([feat, score]) => (
                        <div key={feat} className="importance-item">
                          <span className="feat-label">{feat}</span>
                          <div className="imp-bar"><div className="imp-fill" style={{ width: `${score * 100}%` }}></div></div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="description-card">
                  <div className="desc-header"><ImageIcon size={16} /> <span>Botanical Narrative (Gemini AI Vision)</span></div>
                  <div className="description-text"><ReactMarkdown>{result.description}</ReactMarkdown></div>
                </div>
              </motion.div>
            ) : (
              <div className="empty-state">
                {loading ? <div className="spinner" /> : <ImageIcon size={48} />}
                <p>{loading ? 'Extracting pathogenic features...' : 'Analysis results will appear here'}</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </motion.div>
  );

  return (
    <div className={`App ${darkMode ? 'dark-theme' : 'light-theme'}`}>
      <AnimatePresence mode='wait'>
        {mode === 'landing' && <LandingPage key="landing" />}
        {mode === 'analysis' && <AnalysisPage key="analysis" />}
        {mode === 'gallery' && <GalleryPage key="gallery" />}
      </AnimatePresence>

      <AnimatePresence>
        {selectedReport && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="modal-overlay"
            onClick={() => setSelectedReport(null)}
          >
            <motion.div
              initial={{ y: 50, scale: 0.9 }} animate={{ y: 0, scale: 1 }} exit={{ y: 50, scale: 0.9 }}
              className="modal-content"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="modal-header">
                <h3>{selectedReport.name}</h3>
                <button className="close-btn" onClick={() => setSelectedReport(null)}>&times;</button>
              </div>
              <div className="modal-body markdown-body">
                {reportLoading ? (
                  <div className="modal-loading">
                    <div className="spinner"></div>
                    <p>Loading Technical Data...</p>
                  </div>
                ) : (
                  <>
                    {selectedReport.type === 'markdown' ? (
                      <ReactMarkdown>{reportContent}</ReactMarkdown>
                    ) : selectedReport.type === 'csv' ? (
                      <div className="csv-table-container">
                        <table className="csv-table">
                          {(() => {
                            const lines = reportContent.trim().split('\n');
                            const headers = lines[0].split(',').map(h => h.trim());
                            const rows = lines.slice(1).map(line => line.split(',').map(cell => cell.trim()));

                            return (
                              <>
                                <thead>
                                  <tr>
                                    {headers.map((header, i) => (
                                      <th key={i}>{header}</th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {rows.map((row, i) => (
                                    <tr key={i}>
                                      {row.map((cell, j) => (
                                        <td key={j}>{cell}</td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </>
                            );
                          })()}
                        </table>
                      </div>
                    ) : (
                      <pre className="raw-data-view">
                        {reportContent}
                      </pre>
                    )}
                  </>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
