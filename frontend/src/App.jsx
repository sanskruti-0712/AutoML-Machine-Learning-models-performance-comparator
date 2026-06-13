import React, { useState, useEffect } from 'react';
import { 
  Upload, Settings, Cpu, BarChart3, HelpCircle, FileText, ChevronRight, 
  CheckCircle2, Play, RefreshCw, AlertCircle, Award, Database, 
  ArrowRight, Key, Download, Sparkles, TrendingUp
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, ReferenceLine
} from 'recharts';

const API_BASE = 'http://localhost:8000';

function App() {
  // Navigation
  const [activeStep, setActiveStep] = useState('upload');
  
  // App State
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);
  
  // Upload Data
  const [uploadResult, setUploadResult] = useState(null);
  const [fileName, setFileName] = useState('');
  
  // Preprocessing Settings & Results
  const [targetCol, setTargetCol] = useState('');
  const [problemType, setProblemType] = useState('classification');
  const [scaleData, setScaleData] = useState(true);
  const [detectOutliers, setDetectOutliers] = useState(true);
  const [timeCol, setTimeCol] = useState('');
  const [lags, setLags] = useState(3);
  const [preprocessResult, setPreprocessResult] = useState(null);
  
  // Training Selection & Results
  const [selectedModels, setSelectedModels] = useState([]);
  const [trainResult, setTrainResult] = useState(null);
  
  // Explainability & Tuning State
  const [selectedModel, setSelectedModel] = useState('');
  const [shapData, setShapData] = useState(null);
  const [tuningModel, setTuningModel] = useState('');
  const [nTrials, setNTrials] = useState(10);
  const [tuneResult, setTuneResult] = useState(null);
  
  // AI Advisor
  const [apiKey, setApiKey] = useState('');
  const [showKeyInput, setShowKeyInput] = useState(false);
  const [adviceText, setAdviceText] = useState('');
  const [perspective, setPerspective] = useState('expert');

  // Auto-fill Target Column when file uploads
  useEffect(() => {
    if (uploadResult?.detection) {
      setTargetCol(uploadResult.detection.detected_target || '');
      setProblemType(uploadResult.detection.problem_type || 'classification');
      if (uploadResult.detection.datetime_columns?.length > 0) {
        setTimeCol(uploadResult.detection.datetime_columns[0]);
      }
    }
  }, [uploadResult]);

  // Set default model for SHAP/Tuning once trained
  useEffect(() => {
    if (trainResult?.best_model) {
      setSelectedModel(trainResult.best_model);
      setTuningModel(trainResult.best_model);
      fetchExplainability(trainResult.best_model);
    }
  }, [trainResult]);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setFileName(file.name);
    setError(null);
    setLoading(prev => ({ ...prev, upload: true }));
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to upload dataset.');
      }
      const data = await res.json();
      setUploadResult(data);
      setPreprocessResult(null);
      setTrainResult(null);
      setShapData(null);
      setTuneResult(null);
      setAdviceText('');
      setActiveStep('preprocess');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(prev => ({ ...prev, upload: false }));
    }
  };

  const handlePreprocess = async () => {
    setError(null);
    setLoading(prev => ({ ...prev, preprocess: true }));
    
    try {
      const res = await fetch(`${API_BASE}/api/preprocess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_col: targetCol,
          problem_type: problemType,
          scale_data: scaleData,
          detect_outliers: detectOutliers,
          time_col: problemType === 'time_series' ? timeCol : null,
          lags: parseInt(lags),
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Preprocessing failed.');
      }
      const data = await res.json();
      setPreprocessResult(data);
      
      if (problemType === 'classification') {
        setSelectedModels(['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost', 'LightGBM']);
      } else {
        setSelectedModels(['Linear Regression', 'Ridge Regression', 'Random Forest', 'SVM', 'XGBoost', 'LightGBM']);
      }
      
      setActiveStep('train');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(prev => ({ ...prev, preprocess: false }));
    }
  };

  const handleTrain = async () => {
    setError(null);
    setLoading(prev => ({ ...prev, train: true }));
    
    try {
      const res = await fetch(`${API_BASE}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ selected_models: selectedModels }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Training failed.');
      }
      const data = await res.json();
      setTrainResult(data);
      setActiveStep('dashboard');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(prev => ({ ...prev, train: false }));
    }
  };

  const fetchExplainability = async (modelName) => {
    setLoading(prev => ({ ...prev, explain: true }));
    try {
      const res = await fetch(`${API_BASE}/api/explain?model_name=${encodeURIComponent(modelName)}`);
      if (!res.ok) throw new Error('Failed to fetch SHAP explanation.');
      const data = await res.json();
      setShapData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(prev => ({ ...prev, explain: false }));
    }
  };

  const handleTune = async () => {
    setError(null);
    setLoading(prev => ({ ...prev, tune: true }));
    try {
      const res = await fetch(`${API_BASE}/api/tune`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: tuningModel,
          n_trials: parseInt(nTrials),
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Tuning failed.');
      }
      const data = await res.json();
      setTuneResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(prev => ({ ...prev, tune: false }));
    }
  };

  const handleGetAdvice = async (targetPerspective = perspective) => {
    setError(null);
    setLoading(prev => ({ ...prev, advise: true }));
    try {
      const res = await fetch(`${API_BASE}/api/advise`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          api_key: apiKey || null,
          perspective: targetPerspective 
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to generate AI advice.');
      }
      const data = await res.json();
      setAdviceText(data.advice);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(prev => ({ ...prev, advise: false }));
    }
  };

  const downloadReport = () => {
    window.open(`${API_BASE}/api/report`, '_blank');
  };

  // Model Metric Plot Formatting
  const getChartData = () => {
    if (!trainResult?.results) return [];
    return Object.entries(trainResult.results).map(([name, item]) => {
      const metrics = item.metrics;
      return {
        name,
        score: problemType === 'classification' ? metrics.f1 * 100 : metrics.r2,
        time: metrics.train_time_sec,
        latency: metrics.pred_latency_ms
      };
    }).sort((a, b) => b.score - a.score);
  };

  return (
    <div className="flex min-h-screen bg-slate-50 text-slate-800 selection:bg-blue-500/10 selection:text-blue-700">
      
      {/* SIDEBAR NAVIGATION */}
      <aside className="w-80 border-r border-slate-200 bg-white p-6 flex flex-col justify-between shrink-0">
        <div>
          {/* Brand Logo */}
          <div className="flex items-center gap-3 mb-10">
            <div className="h-9 w-9 rounded-lg bg-blue-600 flex items-center justify-center">
              <Cpu className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-slate-900 leading-tight">AutoML Platform</h1>
              <p className="text-[10px] text-slate-500 font-semibold tracking-wider uppercase">Advisor Dashboard</p>
            </div>
          </div>

          {/* Steps */}
          <nav className="space-y-1">
            {[
              { id: 'upload', label: '1. Upload Dataset', icon: Upload, enabled: true },
              { id: 'preprocess', label: '2. Smart Preprocessing', icon: Settings, enabled: !!uploadResult },
              { id: 'train', label: '3. Train Models', icon: Play, enabled: !!preprocessResult },
              { id: 'dashboard', label: '4. Comparison Dashboard', icon: BarChart3, enabled: !!trainResult },
              { id: 'tune', label: '5. Optuna Tuning', icon: TrendingUp, enabled: !!trainResult },
              { id: 'explain', label: '6. SHAP Explainability', icon: HelpCircle, enabled: !!trainResult },
              { id: 'advise', label: '7. AI-Powered Advisor', icon: Sparkles, enabled: !!trainResult },
            ].map(step => {
              const Icon = step.icon;
              const active = activeStep === step.id;
              return (
                <button
                  key={step.id}
                  disabled={!step.enabled}
                  onClick={() => {
                    setActiveStep(step.id);
                    if (step.id === 'explain' && trainResult?.best_model) {
                      fetchExplainability(selectedModel || trainResult.best_model);
                    }
                  }}
                  className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-semibold transition-all duration-150 ${
                    active 
                      ? 'bg-blue-50 text-blue-700 border-l-2 border-blue-600 pl-2.5' 
                      : step.enabled 
                        ? 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/60' 
                        : 'text-slate-400 cursor-not-allowed'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon className={`h-4 w-4 ${active ? 'text-blue-600' : 'text-slate-400'}`} />
                    <span>{step.label}</span>
                  </div>
                  {step.enabled && step.id !== activeStep && (
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Gemini API Key Panel */}
        <div className="pt-4 border-t border-slate-100">
          <div className="bg-slate-50 rounded-xl p-4 border border-slate-200/60">
            <button 
              onClick={() => setShowKeyInput(!showKeyInput)}
              className="flex items-center justify-between w-full text-xs font-bold text-slate-700 hover:text-blue-600"
            >
              <span className="flex items-center gap-2">
                <Key className="h-3.5 w-3.5 text-slate-400" />
                {apiKey ? 'Gemini Key Configured' : 'Configure Gemini Key'}
              </span>
              <ChevronRight className={`h-3 w-3 transform transition-transform ${showKeyInput ? 'rotate-90' : ''}`} />
            </button>
            
            {showKeyInput && (
              <div className="mt-2.5 space-y-2">
                <p className="text-[10px] text-slate-500 leading-normal">
                  Enter Gemini API key for customized AI advice.
                </p>
                <input
                  type="password"
                  placeholder="AIzaSy..."
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  className="w-full text-xs px-2.5 py-1.5 rounded-lg border border-slate-300 bg-white text-slate-800 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                />
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* MAIN CONTAINER */}
      <main className="flex-1 p-10 overflow-y-auto max-w-5xl mx-auto space-y-8">
        
        {/* Error Callout */}
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3 text-red-800 text-sm">
            <AlertCircle className="h-5 w-5 shrink-0 text-red-600" />
            <div>
              <span className="font-bold">Error:</span> {error}
            </div>
          </div>
        )}

        {/* STEP 1: UPLOAD DATASET */}
        {activeStep === 'upload' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Upload Dataset</h2>
              <p className="text-slate-500 text-sm mt-1">Upload a CSV or Excel file to begin model training and evaluation.</p>
            </div>
            
            {/* Upload Zone */}
            <div className="minimal-card rounded-2xl p-10 text-center border-dashed border-slate-300 hover:border-blue-500 bg-white transition-all duration-200 flex flex-col items-center">
              <div className="h-12 w-12 rounded-xl bg-blue-50 flex items-center justify-center mb-3">
                <Upload className="h-5 w-5 text-blue-600" />
              </div>
              <p className="text-sm font-bold text-slate-800">Drag and drop your dataset file here</p>
              <p className="text-xs text-slate-500 mt-1">CSV, XLSX or XLS formats up to 50MB</p>
              
              <label className="mt-5 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-bold transition-all cursor-pointer shadow-sm">
                {loading.upload ? 'Uploading...' : 'Select File'}
                <input type="file" accept=".csv,.xlsx,.xls" onChange={handleFileUpload} className="hidden" disabled={loading.upload} />
              </label>
            </div>

            {/* If uploaded */}
            {uploadResult && (
              <div className="minimal-card rounded-2xl p-6 space-y-6">
                <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                  <div className="flex items-center gap-2.5">
                    <Database className="h-4.5 w-4.5 text-slate-500" />
                    <span className="font-bold text-slate-900 text-base">{fileName}</span>
                  </div>
                  <div className="px-2.5 py-0.5 bg-blue-50 border border-blue-200 rounded-full text-[10px] font-bold text-blue-700">
                    Auto-Detected Target: {uploadResult.detection.detected_target}
                  </div>
                </div>

                {/* Data Stats Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[
                    { label: 'Total Rows', val: uploadResult.total_rows.toLocaleString() },
                    { label: 'Total Features', val: (uploadResult.total_cols - 1).toLocaleString() },
                    { label: 'Problem Type', val: uploadResult.detection.problem_type.toUpperCase() },
                    { label: 'Target Cardinality', val: uploadResult.detection.unique_target_values },
                  ].map((stat, i) => (
                    <div key={i} className="bg-slate-50 p-4 rounded-xl border border-slate-100">
                      <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">{stat.label}</p>
                      <p className="text-lg font-bold text-slate-800 mt-1">{stat.val}</p>
                    </div>
                  ))}
                </div>

                {/* Preview Table */}
                <div className="space-y-2">
                  <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">Dataset Preview</h3>
                  <div className="overflow-x-auto rounded-lg border border-slate-200">
                    <table className="w-full text-left text-xs bg-white">
                      <thead>
                        <tr className="bg-slate-50 text-slate-600 border-b border-slate-200 font-semibold">
                          {Object.keys(uploadResult.sample_data[0] || {}).map(key => (
                            <th key={key} className="px-4 py-2.5 border-r border-slate-200 last:border-0">{key}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 text-slate-700">
                        {uploadResult.sample_data.slice(0, 5).map((row, i) => (
                          <tr key={i} className="hover:bg-slate-50/50">
                            {Object.values(row).map((val, idx) => (
                              <td key={idx} className="px-4 py-2.5 truncate max-w-[120px] border-r border-slate-150 last:border-0">
                                {val === null ? <span className="text-slate-400">NaN</span> : String(val)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="flex justify-end pt-2 border-t border-slate-100">
                  <button
                    onClick={() => setActiveStep('preprocess')}
                    className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-bold shadow-sm"
                  >
                    Configure Preprocessing <ArrowRight className="h-4 w-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* STEP 2: PREPROCESSING CONFIGURATION */}
        {activeStep === 'preprocess' && uploadResult && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Smart Preprocessing</h2>
              <p className="text-slate-500 text-sm mt-1">Configure data cleanup settings and feature scaling inputs.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              
              {/* Settings Box */}
              <div className="md:col-span-2 minimal-card rounded-2xl p-6 space-y-6 bg-white">
                <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2 pb-3 border-b border-slate-100">
                  <Settings className="h-4 w-4 text-blue-600" /> Pipeline Settings
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Select Target */}
                  <div className="space-y-1">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Target Column</label>
                    <select
                      value={targetCol}
                      onChange={(e) => setTargetCol(e.target.value)}
                      className="w-full text-xs px-3 py-2 rounded-lg bg-white border border-slate-300 text-slate-800 focus:outline-none focus:border-blue-500"
                    >
                      {uploadResult.columns.map(col => (
                        <option key={col.name} value={col.name}>{col.name} ({col.type})</option>
                      ))}
                    </select>
                  </div>

                  {/* Select Problem Type */}
                  <div className="space-y-1">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Problem Type</label>
                    <select
                      value={problemType}
                      onChange={(e) => setProblemType(e.target.value)}
                      className="w-full text-xs px-3 py-2 rounded-lg bg-white border border-slate-300 text-slate-800 focus:outline-none focus:border-blue-500"
                    >
                      <option value="classification">Classification (Categorical Target)</option>
                      <option value="regression">Regression (Continuous Target)</option>
                      <option value="time_series">Time Series Regression (Auto-Lag)</option>
                    </select>
                  </div>

                  {/* Time Series Specific Options */}
                  {problemType === 'time_series' && (
                    <>
                      <div className="space-y-1">
                        <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">DateTime Column</label>
                        <select
                          value={timeCol}
                          onChange={(e) => setTimeCol(e.target.value)}
                          className="w-full text-xs px-3 py-2 rounded-lg bg-white border border-slate-300 text-slate-800 focus:outline-none focus:border-blue-500"
                        >
                          <option value="">-- Choose DateTime Column --</option>
                          {uploadResult.columns.map(col => (
                            <option key={col.name} value={col.name}>{col.name}</option>
                          ))}
                        </select>
                      </div>

                      <div className="space-y-1">
                        <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Number of Target Lags</label>
                        <input
                          type="number"
                          min="1"
                          max="10"
                          value={lags}
                          onChange={(e) => setLags(e.target.value)}
                          className="w-full text-xs px-3 py-2 rounded-lg bg-white border border-slate-300 text-slate-800 focus:outline-none focus:border-blue-500"
                        />
                      </div>
                    </>
                  )}

                  {/* Global settings */}
                  <div className="space-y-3 md:col-span-2 pt-2">
                    <label className="flex items-start gap-2.5 cursor-pointer">
                      <input 
                        type="checkbox" 
                        checked={scaleData} 
                        onChange={(e) => setScaleData(e.target.checked)}
                        className="mt-0.5 h-4 w-4 border-slate-300 rounded text-blue-600 focus:ring-blue-500/20"
                      />
                      <div>
                        <p className="text-xs font-bold text-slate-800">Apply Standard Feature Scaling</p>
                        <p className="text-[10px] text-slate-500">Standardizes numerical properties (mean=0, std=1).</p>
                      </div>
                    </label>

                    <label className="flex items-start gap-2.5 cursor-pointer">
                      <input 
                        type="checkbox" 
                        checked={detectOutliers} 
                        onChange={(e) => setDetectOutliers(e.target.checked)}
                        className="mt-0.5 h-4 w-4 border-slate-300 rounded text-blue-600 focus:ring-blue-500/20"
                      />
                      <div>
                        <p className="text-xs font-bold text-slate-800">Filter Outliers via Isolation Forest</p>
                        <p className="text-[10px] text-slate-500">Filters anomalous rows using multivariate Isolation Forest (3%).</p>
                      </div>
                    </label>
                  </div>
                </div>

                <div className="flex justify-end pt-4 border-t border-slate-100">
                  <button
                    onClick={handlePreprocess}
                    disabled={loading.preprocess}
                    className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-bold shadow-sm"
                  >
                    {loading.preprocess ? 'Running Preprocessing...' : 'Run Pipeline'}
                  </button>
                </div>
              </div>

              {/* Data Summary Panel */}
              <div className="space-y-6">
                <div className="minimal-card rounded-2xl p-5 bg-white">
                  <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-3 border-b border-slate-100 pb-2">Columns List</h3>
                  <div className="space-y-2.5 max-h-[260px] overflow-y-auto pr-1 text-[11px]">
                    {uploadResult.columns.map((col, idx) => (
                      <div key={idx} className="flex justify-between items-center py-1 border-b border-slate-50 last:border-0">
                        <div>
                          <span className="font-semibold text-slate-700">{col.name}</span>
                          <span className="ml-1.5 text-[9px] text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded-full">{col.type}</span>
                        </div>
                        <div className="text-slate-400 font-medium">
                          {col.null_count > 0 ? (
                            <span className="text-amber-600 font-semibold">{col.null_count} nulls</span>
                          ) : (
                            <span>OK</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Preprocess results */}
                {preprocessResult && (
                  <div className="minimal-card rounded-2xl p-5 bg-white border-l-4 border-l-emerald-500">
                    <h3 className="text-xs font-bold text-emerald-700 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                      <CheckCircle2 className="h-4 w-4 text-emerald-600" /> Pipeline Processed
                    </h3>
                    <p className="text-[11px] text-slate-500">Cleaned Shape: <strong className="text-slate-800">{preprocessResult.preprocessed_shape[0]} rows × {preprocessResult.preprocessed_shape[1]} columns</strong></p>
                    
                    {/* Preprocessing log */}
                    <div className="mt-3.5 space-y-2">
                      <p className="text-[9px] font-bold text-slate-400 uppercase tracking-wider">Executed Stages:</p>
                      <div className="space-y-1.5 max-h-[160px] overflow-y-auto pr-1">
                        {preprocessResult.steps_log.map((step, i) => (
                          <div key={i} className="p-2 bg-slate-50 rounded-lg text-[10px] border border-slate-100">
                            <strong className="text-blue-700 block">{step.step}</strong>
                            <span className="text-slate-500 mt-0.5 block">{step.details}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>

            </div>
          </div>
        )}

        {/* STEP 3: MODEL TRAINING */}
        {activeStep === 'train' && preprocessResult && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Train & Evaluate Models</h2>
              <p className="text-slate-500 text-sm mt-1">Select the machine learning models to run and evaluate against each other.</p>
            </div>

            <div className="minimal-card rounded-2xl p-6 space-y-6 max-w-xl bg-white">
              <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2 pb-3 border-b border-slate-100">
                <Play className="h-4 w-4 text-blue-600" /> Choose Models
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {(problemType === 'classification' 
                  ? ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost', 'LightGBM'] 
                  : ['Linear Regression', 'Ridge Regression', 'Random Forest', 'SVM', 'XGBoost', 'LightGBM']
                ).map(modelName => {
                  const checked = selectedModels.includes(modelName);
                  return (
                    <label 
                      key={modelName}
                      className={`flex items-center gap-2.5 p-3 rounded-xl border transition-all cursor-pointer text-xs ${
                        checked 
                          ? 'bg-blue-50/50 border-blue-200 text-blue-900 font-bold' 
                          : 'bg-white border-slate-200 text-slate-600 hover:border-slate-300'
                      }`}
                    >
                      <input 
                        type="checkbox"
                        checked={checked}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedModels([...selectedModels, modelName]);
                          } else {
                            setSelectedModels(selectedModels.filter(m => m !== modelName));
                          }
                        }}
                        className="h-4 w-4 border-slate-350 text-blue-600 focus:ring-blue-500/10"
                      />
                      <span>{modelName}</span>
                    </label>
                  );
                })}
              </div>

              <div className="flex justify-end pt-4 border-t border-slate-100">
                <button
                  onClick={handleTrain}
                  disabled={loading.train || selectedModels.length === 0}
                  className="flex items-center gap-1.5 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-100 disabled:text-slate-400 rounded-lg text-xs font-bold text-white shadow-sm transition-all"
                >
                  {loading.train ? 'Training models...' : 'Run Training & Evaluation'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* STEP 4: DASHBOARD RESULTS */}
        {activeStep === 'dashboard' && trainResult && (
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Performance Dashboard</h2>
                <p className="text-slate-500 text-sm mt-1">Summary metrics of model performance comparisons.</p>
              </div>
              <div className="flex items-center gap-3">
                <div className="bg-emerald-50 border border-emerald-200 px-4 py-2.5 rounded-xl flex items-center gap-2">
                  <Award className="h-4.5 w-4.5 text-emerald-600" />
                  <div>
                    <span className="text-[9px] text-slate-500 font-bold uppercase tracking-wider block">Best Model</span>
                    <span className="text-xs font-bold text-emerald-800">{trainResult.best_model}</span>
                  </div>
                </div>
                <button
                  onClick={() => setActiveStep('tune')}
                  className="flex items-center gap-1.5 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-bold shadow-sm transition-all"
                >
                  Tune Best Model <ArrowRight className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>

            {/* Performance Summary Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              
              {/* Score Chart */}
              <div className="minimal-card rounded-2xl p-5 bg-white">
                <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-4">
                  {problemType === 'classification' ? 'F1-Score (%)' : 'R² Score (Goodness of Fit)'}
                </h3>
                <div className="h-60">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getChartData()} margin={{ top: 10, right: 10, left: -25, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis dataKey="name" stroke="#94a3b8" fontSize={9} fontClass="font-semibold" />
                      <YAxis stroke="#94a3b8" fontSize={9} domain={problemType === 'classification' ? [0, 100] : [0, 'auto']} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '8px' }}
                        itemStyle={{ color: '#2563eb' }}
                      />
                      <Bar dataKey="score" fill="#2563eb" radius={[4, 4, 0, 0]} maxBarSize={40} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Training Time Chart */}
              <div className="minimal-card rounded-2xl p-5 bg-white">
                <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-4">Training Time (Seconds)</h3>
                <div className="h-60">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getChartData()} margin={{ top: 10, right: 10, left: -25, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis dataKey="name" stroke="#94a3b8" fontSize={9} />
                      <YAxis stroke="#94a3b8" fontSize={9} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '8px' }}
                        itemStyle={{ color: '#475569' }}
                      />
                      <Bar dataKey="time" fill="#475569" radius={[4, 4, 0, 0]} maxBarSize={40} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

            </div>

            {/* Performance Summary Table */}
            <div className="minimal-card rounded-2xl p-5 bg-white">
              <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-4">Comparison Table</h3>
              <div className="overflow-x-auto rounded-lg border border-slate-200">
                <table className="w-full text-left text-xs bg-white">
                  <thead>
                    <tr className="bg-slate-50 text-slate-600 border-b border-slate-200 font-semibold">
                      <th className="px-5 py-3">Model Name</th>
                      {problemType === 'classification' ? (
                        <>
                          <th className="px-4 py-3">Accuracy</th>
                          <th className="px-4 py-3">F1-Score</th>
                          <th className="px-4 py-3">Precision</th>
                          <th className="px-4 py-3">Recall</th>
                        </>
                      ) : (
                        <>
                          <th className="px-4 py-3">R² Score</th>
                          <th className="px-4 py-3">MAE</th>
                          <th className="px-4 py-3">RMSE</th>
                        </>
                      )}
                      <th className="px-4 py-3">Train Time</th>
                      <th className="px-4 py-3">Latency</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-slate-700">
                    {Object.entries(trainResult.results).map(([name, item]) => {
                      const m = item.metrics;
                      const isBest = name === trainResult.best_model;
                      return (
                        <tr key={name} className={`hover:bg-slate-50/50 ${isBest ? 'bg-blue-50/40 text-blue-900 font-bold' : ''}`}>
                          <td className="px-5 py-3 flex items-center gap-2 border-r border-slate-100">
                            {name} 
                            {isBest && <span className="text-[8px] bg-emerald-50 text-emerald-700 border border-emerald-200 px-1.5 py-0.5 rounded-full font-bold">Best</span>}
                          </td>
                          {problemType === 'classification' ? (
                            <>
                              <td className="px-4 py-3 border-r border-slate-100">{(m.accuracy * 100).toFixed(2)}%</td>
                              <td className="px-4 py-3 border-r border-slate-100">{(m.f1 * 100).toFixed(2)}%</td>
                              <td className="px-4 py-3 border-r border-slate-100">{(m.precision * 100).toFixed(2)}%</td>
                              <td className="px-4 py-3 border-r border-slate-100">{(m.recall * 100).toFixed(2)}%</td>
                            </>
                          ) : (
                            <>
                              <td className="px-4 py-3 border-r border-slate-100">{m.r2.toFixed(4)}</td>
                              <td className="px-4 py-3 border-r border-slate-100">{m.mae.toFixed(4)}</td>
                              <td className="px-4 py-3 border-r border-slate-100">{m.rmse.toFixed(4)}</td>
                            </>
                          )}
                          <td className="px-4 py-3 border-r border-slate-100">{m.train_time_sec.toFixed(3)}s</td>
                          <td className="px-4 py-3">{(m.pred_latency_ms).toFixed(4)} ms</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Extra Confusion Matrix for Classification */}
            {problemType === 'classification' && trainResult.results[trainResult.best_model]?.metrics?.confusion_matrix && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* Confusion Matrix Plot */}
                <div className="minimal-card rounded-2xl p-5 bg-white">
                  <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-4">Confusion Matrix ({trainResult.best_model})</h3>
                  <div className="flex items-center justify-center h-48">
                    <div className="grid grid-cols-2 gap-3 w-64">
                      {trainResult.results[trainResult.best_model].metrics.confusion_matrix.map((row, rIdx) => 
                        row.map((val, cIdx) => {
                          const isDiagonal = rIdx === cIdx;
                          return (
                            <div 
                              key={`${rIdx}-${cIdx}`} 
                              className={`flex flex-col items-center justify-center rounded-xl border p-3.5 ${
                                isDiagonal 
                                  ? 'bg-blue-50 border-blue-200 text-blue-800' 
                                  : 'bg-slate-50 border-slate-200 text-slate-500'
                              }`}
                            >
                              <span className="text-[9px] text-slate-400 font-bold uppercase">
                                {rIdx === 0 ? 'Act Neg' : 'Act Pos'} &rarr; {cIdx === 0 ? 'Pred Neg' : 'Pred Pos'}
                              </span>
                              <span className="text-lg font-bold mt-1 text-slate-800">{val}</span>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="minimal-card rounded-2xl p-5 bg-white flex flex-col justify-center">
                  <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">Metrics Summary</h4>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    Estimators were validated using stratified cross-validation. The winning model **{trainResult.best_model}** exhibited optimal F1 metrics of 
                    **{(trainResult.results[trainResult.best_model].metrics.f1 * 100).toFixed(2)}%**. You can further analyze and tune this model's hyperparameters using the tuning panel.
                  </p>
                </div>

              </div>
            )}
          </div>
        )}

        {/* STEP 5: HYPERPARAMETER TUNING */}
        {activeStep === 'tune' && trainResult && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Hyperparameter Tuning</h2>
              <p className="text-slate-500 text-sm mt-1">Tune model parameters via Optuna algorithms to optimize goodness of fit metrics.</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* Controls */}
              <div className="minimal-card rounded-2xl p-5 bg-white space-y-4">
                <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider pb-2 border-b border-slate-100">Configuration</h3>
                
                {/* Select Model */}
                <div className="space-y-1">
                  <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Select Model</label>
                  <select
                    value={tuningModel}
                    onChange={(e) => setTuningModel(e.target.value)}
                    className="w-full text-xs px-3 py-2 rounded-lg bg-white border border-slate-300 text-slate-800 focus:outline-none focus:border-blue-500"
                  >
                    {Object.keys(trainResult.results).map(name => (
                      <option key={name} value={name}>{name}</option>
                    ))}
                  </select>
                </div>

                {/* Trials */}
                <div className="space-y-1">
                  <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Optuna Trials</label>
                  <input
                    type="number"
                    min="5"
                    max="50"
                    value={nTrials}
                    onChange={(e) => setNTrials(e.target.value)}
                    className="w-full text-xs px-3 py-2 rounded-lg bg-white border border-slate-300 text-slate-800 focus:outline-none focus:border-blue-500"
                  />
                </div>

                <button
                  onClick={handleTune}
                  disabled={loading.tune}
                  className="w-full py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-1.5 shadow-sm"
                >
                  {loading.tune ? (
                    <>
                      <RefreshCw className="h-3.5 w-3.5 animate-spin" /> Tuning...
                    </>
                  ) : (
                    <>
                      <TrendingUp className="h-3.5 w-3.5" /> Optimize Parameters
                    </>
                  )}
                </button>
              </div>

              {/* Tuning Results */}
              <div className="lg:col-span-2 space-y-6">
                {tuneResult ? (
                  <div className="space-y-6">
                    {/* Before vs After Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      
                      <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl">
                        <span className="text-[9px] text-slate-500 font-bold uppercase tracking-wider">Baseline Score</span>
                        <div className="flex items-baseline gap-1.5 mt-1">
                          <span className="text-2xl font-bold text-slate-700">
                            {tuneResult.metric_name === 'f1_weighted' ? `${(tuneResult.before_score * 100).toFixed(1)}%` : tuneResult.before_score.toFixed(4)}
                          </span>
                          <span className="text-[10px] text-slate-450">Cross-Validated</span>
                        </div>
                      </div>

                      <div className="bg-blue-50 border border-blue-200 p-4 rounded-xl">
                        <span className="text-[9px] text-blue-600 font-bold uppercase tracking-wider">Optimized Score</span>
                        <div className="flex items-baseline gap-1.5 mt-1">
                          <span className="text-2xl font-bold text-blue-800">
                            {tuneResult.metric_name === 'f1_weighted' ? `${(tuneResult.after_score * 100).toFixed(1)}%` : tuneResult.after_score.toFixed(4)}
                          </span>
                          <span className="text-[10px] font-bold text-blue-600">
                            {tuneResult.improvement > 0 ? `+${(tuneResult.improvement * (tuneResult.metric_name === 'f1_weighted' ? 100 : 1)).toFixed(1)}%` : 'No change'}
                          </span>
                        </div>
                      </div>

                    </div>

                    {/* Convergence chart */}
                    <div className="minimal-card rounded-2xl p-5 bg-white">
                      <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-4">Optuna Trial History</h3>
                      <div className="h-52">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={tuneResult.trial_history} margin={{ top: 10, right: 10, left: -25, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                            <XAxis dataKey="trial_number" stroke="#94a3b8" fontSize={9} />
                            <YAxis stroke="#94a3b8" fontSize={9} />
                            <Tooltip contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0' }} />
                            <Line type="monotone" dataKey="score" stroke="#2563eb" strokeWidth={1.5} dot={{ r: 3 }} activeDot={{ r: 5 }} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* Best Parameters */}
                    <div className="minimal-card rounded-2xl p-5 bg-white">
                      <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-3">Best Parameters Map</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {Object.entries(tuneResult.best_params).map(([key, val]) => (
                          <div key={key} className="flex justify-between items-center bg-slate-50 border border-slate-100 p-3 rounded-lg">
                            <span className="text-xs text-slate-500 font-mono">{key}</span>
                            <span className="text-xs font-bold text-blue-700 font-mono">
                              {typeof val === 'number' && !Number.isInteger(val) ? val.toFixed(5) : String(val)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>

                  </div>
                ) : (
                  <div className="minimal-card rounded-2xl p-10 text-center text-slate-400 bg-white">
                    <TrendingUp className="h-8 w-8 text-slate-350 mx-auto mb-2" />
                    <p className="text-xs font-semibold">Select a model and run optimization to display studies.</p>
                  </div>
                )}
              </div>

            </div>
          </div>
        )}

        {/* STEP 6: EXPLAINABILITY (SHAP / FEATURE IMPORTANCE) */}
        {activeStep === 'explain' && trainResult && (
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h2 className="text-2xl font-bold text-slate-900 tracking-tight">SHAP Explainability</h2>
                <p className="text-slate-500 text-sm mt-1">Quantify and trace the influence of input features on trained estimators.</p>
              </div>
              <div>
                <select
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value);
                    fetchExplainability(e.target.value);
                  }}
                  className="px-3 py-1.5 rounded-lg bg-white border border-slate-350 text-slate-700 focus:outline-none focus:border-blue-500 text-xs font-semibold"
                >
                  {Object.keys(trainResult.results).map(name => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
              </div>
            </div>

            {loading.explain ? (
              <div className="minimal-card rounded-2xl p-20 text-center bg-white flex flex-col items-center">
                <RefreshCw className="h-8 w-8 text-blue-600 animate-spin mb-3" />
                <p className="text-xs font-semibold text-slate-600">Computing SHAP values (permuting features)...</p>
              </div>
            ) : shapData ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                
                {/* Global SHAP Bar chart */}
                <div className="minimal-card rounded-2xl p-5 bg-white">
                  <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-4 font-semibold">Global Feature Contribution (Mean |SHAP Value|)</h3>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart 
                        data={shapData.global_shap}
                        layout="vertical"
                        margin={{ top: 10, right: 10, left: 15, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis type="number" stroke="#94a3b8" fontSize={9} />
                        <YAxis dataKey="feature" type="category" stroke="#94a3b8" fontSize={9} width={70} />
                        <Tooltip contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0' }} />
                        <Bar dataKey="shap_value" fill="#2563eb" radius={[0, 4, 4, 0]} maxBarSize={20} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Local Explanation Waterfall */}
                <div className="minimal-card rounded-2xl p-5 bg-white">
                  <div className="mb-4 pb-2 border-b border-slate-100">
                    <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider font-semibold">Waterfall Contribution Chart</h3>
                    <p className="text-[10px] text-slate-400 mt-0.5">Explaining sample prediction push (Base value: {shapData.base_value.toFixed(3)})</p>
                  </div>
                  <div className="h-64 overflow-y-auto space-y-2.5 pr-1 text-xs">
                    {shapData.local_explanation.map((item, idx) => {
                      const positive = item.shap_value >= 0;
                      return (
                        <div key={idx} className="bg-slate-50 border border-slate-200/60 p-2.5 rounded-lg flex justify-between items-center">
                          <div className="space-y-1">
                            <div className="flex items-center gap-1.5">
                              <span className="font-bold text-slate-800">{item.feature}</span>
                              <span className="text-[9px] text-slate-450 font-mono bg-white px-1.5 py-0.5 border border-slate-200 rounded">val: {item.actual_value}</span>
                            </div>
                            <div className="h-1 w-32 bg-slate-200 rounded-full overflow-hidden">
                              <div 
                                className={`h-full ${positive ? 'bg-blue-600' : 'bg-rose-500'}`}
                                style={{ width: `${Math.min(100, Math.abs(item.shap_value) * 300)}%` }}
                              />
                            </div>
                          </div>
                          <div className="text-right font-mono text-[11px]">
                            <span className={`font-bold ${positive ? 'text-blue-600' : 'text-rose-600'}`}>
                              {positive ? `+${item.shap_value.toFixed(4)}` : item.shap_value.toFixed(4)}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

              </div>
            ) : (
              <div className="minimal-card rounded-2xl p-10 text-center text-slate-400 bg-white">
                <HelpCircle className="h-8 w-8 text-slate-350 mx-auto mb-2" />
                <p className="text-xs font-semibold">No explainability calculations loaded.</p>
              </div>
            )}
          </div>
        )}

        {/* STEP 7: AI ADVISOR */}
        {activeStep === 'advise' && trainResult && (
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h2 className="text-2xl font-bold text-slate-900 tracking-tight">AI Advisor Recommendations</h2>
                <p className="text-slate-500 text-sm mt-1">Triggers rule-based heuristics or Gemini models to explain model victories.</p>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                {/* Perspective Selector */}
                <div className="flex bg-slate-200/50 p-1 rounded-xl text-[10px] font-bold shrink-0 border border-slate-200/60">
                  <button
                    onClick={() => {
                      setPerspective('expert');
                      if (adviceText) handleGetAdvice('expert');
                    }}
                    className={`px-2.5 py-1 rounded-lg transition-all ${
                      perspective === 'expert'
                        ? 'bg-white text-slate-800 shadow-sm'
                        : 'text-slate-500 hover:text-slate-800'
                    }`}
                  >
                    Expert Scientist
                  </button>
                  <button
                    onClick={() => {
                      setPerspective('layman');
                      if (adviceText) handleGetAdvice('layman');
                    }}
                    className={`px-2.5 py-1 rounded-lg transition-all ${
                      perspective === 'layman'
                        ? 'bg-white text-blue-700 shadow-sm'
                        : 'text-slate-500 hover:text-slate-800'
                    }`}
                  >
                    Simplified Layman
                  </button>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => handleGetAdvice(perspective)}
                    disabled={loading.advise}
                    className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-bold shadow-sm transition-all"
                  >
                    {loading.advise ? (
                      <>
                        <RefreshCw className="h-3.5 w-3.5 animate-spin" /> Generating...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-3.5 w-3.5" /> {apiKey ? 'Query Gemini Advisor' : 'Generate Advisor Summary'}
                      </>
                    )}
                  </button>
                  <button
                    onClick={downloadReport}
                    className="flex items-center gap-1.5 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-bold shadow-sm transition-all"
                  >
                    <Download className="h-3.5 w-3.5" /> Export PDF
                  </button>
                </div>
              </div>
            </div>

            {adviceText ? (
              <div className="minimal-card rounded-2xl p-6 space-y-5 bg-white border-l-4 border-l-blue-500">
                <div className="flex items-center gap-2 text-slate-800 font-bold text-base pb-3 border-b border-slate-100">
                  <Sparkles className="h-4.5 w-4.5 text-blue-600" />
                  <span>Advisor Transcript</span>
                </div>
                
                <div className="text-slate-650 leading-relaxed text-xs space-y-3.5 font-normal max-h-[440px] overflow-y-auto pr-1">
                  {adviceText.split('\n').map((line, idx) => {
                    const text = line.trim();
                    if (!text) return null;
                    
                    if (text.startsWith('###')) {
                      return <h4 key={idx} className="text-sm font-bold text-slate-900 mt-4 mb-1.5 uppercase tracking-wider">{text.replace('###', '').trim()}</h4>;
                    }
                    if (text.startsWith('####')) {
                      return <h5 key={idx} className="text-xs font-bold text-blue-700 mt-3 mb-1.5">{text.replace('####', '').trim()}</h5>;
                    }
                    if (text.startsWith('-') || text.startsWith('*')) {
                      return (
                        <div key={idx} className="flex items-start gap-2 ml-3 my-0.5">
                          <span className="text-blue-500 mt-1 shrink-0 h-1 w-1 rounded-full bg-blue-600" />
                          <span>{text.substring(1).trim()}</span>
                        </div>
                      );
                    }
                    if (text.match(/^\d+\./)) {
                      return (
                        <div key={idx} className="flex items-start gap-2 ml-3 my-1">
                          <span className="text-blue-600 font-bold font-mono">{text.match(/^\d+\./)[0]}</span>
                          <span>{text.replace(/^\d+\./, '').trim()}</span>
                        </div>
                      );
                    }
                    return <p key={idx} className="my-1.5">{text}</p>;
                  })}
                </div>
              </div>
            ) : (
              <div className="minimal-card rounded-2xl p-12 text-center text-slate-400 bg-white">
                <Sparkles className="h-8 w-8 text-slate-350 mx-auto mb-2" />
                <p className="text-xs font-semibold">Click "Generate Advisor Summary" to load metrics analysis.</p>
              </div>
            )}
          </div>
        )}

      </main>
    </div>
  );
}

export default App;
