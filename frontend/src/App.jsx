import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Send, Upload, Loader2, Bot, User, FileText, Trash2, Globe, FileOutput, Clock, History, Volume2, StopCircle, Plus } from 'lucide-react';

// ⚠️ YOUR RAILWAY URL
const API_BASE_URL = "https://ai-pdf-rag-production.up.railway.app"; 

function App() {
  const [file, setFile] = useState(null);
  const [pdfUrl, setPdfUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [documents, setDocuments] = useState([]); 
  
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [thinking, setThinking] = useState(false);
  
  // --- AUDIO STATE ---
  const [isSpeaking, setIsSpeaking] = useState(false);

  // --- 1. LOAD HISTORY ---
  useEffect(() => {
    fetchHistory();
    fetchDocuments();
    return () => window.speechSynthesis.cancel();
  }, []);

  const fetchHistory = async () => {
    try {
        const res = await axios.get(`${API_BASE_URL}/history`);
        if (res.data.history.length > 0) {
            setMessages(res.data.history);
        } else {
            setMessages([{ type: 'ai', text: "Hello! Upload a PDF to start. I recall our past chats!" }]);
        }
    } catch (e) { console.error("History Error", e); }
  };

  const fetchDocuments = async () => {
    try {
        const res = await axios.get(`${API_BASE_URL}/documents`);
        setDocuments(res.data.documents);
    } catch (e) { console.error("Docs Error", e); }
  };

  // --- 2. AUDIO LOGIC ---
  const speakText = (text) => {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US'; 
        utterance.rate = 1; 
        utterance.onstart = () => setIsSpeaking(true);
        utterance.onend = () => setIsSpeaking(false);
        utterance.onerror = () => setIsSpeaking(false);
        window.speechSynthesis.speak(utterance);
    } else {
        alert("Sorry, your browser doesn't support Text-to-Speech!");
    }
  };

  const stopSpeaking = () => {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
  };

  // --- 3. RESET & NEW UPLOAD ---
  
  // A. Destructive Reset (Deletes DB)
  const handleClearHistory = async () => {
    if(!window.confirm("Are you sure? This deletes ALL chat history and saved files.")) return;
    stopSpeaking();
    try {
        await axios.delete(`${API_BASE_URL}/clear`);
        setFile(null);
        setPdfUrl(null);
        setMessages([{ type: 'ai', text: "History cleared. Ready for a new document!" }]);
        fetchDocuments(); 
    } catch(e) { alert("Error clearing history"); }
  };

  // B. New Upload (UI Reset Only)
  const handleNewUpload = () => {
      stopSpeaking();
      setPdfUrl(null); // This hides the PDF viewer and shows the Upload box
      setFile(null);
      // Optional: Add a separator line in chat or just keep history visible
      setMessages(prev => [...prev, { type: 'ai', text: "--- Ready for new file ---" }]);
  };

  // --- 4. UPLOAD ---
  const handleFileChange = async (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      await handleUpload(selectedFile);
    }
  };

  const handleUpload = async (selectedFile) => {
    setUploading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      
      const url = `${API_BASE_URL}/static/${selectedFile.name}`;
      setPdfUrl(url);
      setMessages(prev => [...prev, { type: 'ai', text: `I've read ${selectedFile.name}. Ask me anything!` }]);
      fetchDocuments();
    } catch (error) {
      console.error("Upload Error:", error);
      alert("Error uploading file.");
    } finally {
      setUploading(false);
    }
  };

  const loadOldFile = (filename) => {
      setPdfUrl(`${API_BASE_URL}/static/${filename}`);
      setMessages(prev => [...prev, { type: 'ai', text: `Loaded ${filename} from history.` }]);
  };

  // --- 5. CHAT ---
  const handleAsk = async (customQuestion = null) => {
    const textToSend = customQuestion || question;
    if (!textToSend.trim()) return;

    if (!customQuestion) {
        setMessages(prev => [...prev, { type: 'user', text: textToSend }]);
    }
    
    setQuestion("");
    setThinking(true);

    try {
      const res = await axios.post(`${API_BASE_URL}/query`, { question: textToSend });
      setMessages(prev => [...prev, { type: 'ai', text: res.data.answer }]);
    } catch (error) {
      setMessages(prev => [...prev, { type: 'ai', text: "Error: Could not get answer." }]);
    } finally {
      setThinking(false);
    }
  };

  const handleSummary = async () => {
    setMessages(prev => [...prev, { type: 'user', text: "✨ Generating Document Summary..." }]);
    setThinking(true);
    try {
        const res = await axios.post(`${API_BASE_URL}/summarize`);
        setMessages(prev => [...prev, { type: 'ai', text: res.data.summary }]);
    } catch (error) {
        setMessages(prev => [...prev, { type: 'ai', text: "Error generating summary." }]);
    } finally {
        setThinking(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-900 text-white font-sans overflow-hidden">
      
      {/* LEFT SIDE: PDF Viewer & History */}
      <div className="w-1/2 border-r border-gray-700 bg-gray-800 flex flex-col">
        
        {/* Header */}
        <div className="p-4 border-b border-gray-700 flex justify-between items-center bg-gray-900">
          <h1 className="text-xl font-bold text-blue-400 flex items-center">
            <FileText className="mr-2" /> Document Viewer
          </h1>
          <div className="flex space-x-2">
            {/* NEW UPLOAD BUTTON */}
            <button onClick={handleNewUpload} className="bg-blue-600 hover:bg-blue-500 text-white px-3 py-1 rounded text-sm flex items-center transition" title="Upload a different file">
                <Plus size={16} className="mr-1"/> New
            </button>
            
            {pdfUrl && (
                <button onClick={handleSummary} disabled={thinking} className="bg-green-600 hover:bg-green-500 text-white px-3 py-1 rounded text-sm flex items-center transition">
                    <FileOutput size={16} className="mr-1"/> Summary
                </button>
            )}
            
            <button onClick={handleClearHistory} className="bg-red-600 hover:bg-red-500 text-white px-3 py-1 rounded text-sm flex items-center transition" title="Delete all history">
                <Trash2 size={16} className="mr-1"/> Clear
            </button>
          </div>
        </div>

        {/* PDF Content */}
        <div className="flex-1 bg-gray-700 relative overflow-y-auto">
          {pdfUrl ? (
            <iframe src={pdfUrl} className="w-full h-full border-none" title="PDF Viewer" />
          ) : (
            <div className="p-8">
                {/* Upload Box */}
                <div className="mb-8 flex justify-center">
                    <label className="border-2 border-dashed border-gray-500 rounded-xl p-8 hover:border-blue-500 hover:bg-gray-700/50 transition cursor-pointer group w-full max-w-md text-center">
                        <input type="file" onChange={handleFileChange} className="hidden" accept=".pdf" />
                        <div className="flex flex-col items-center">
                        <div className="w-12 h-12 bg-gray-800 rounded-full flex items-center justify-center mb-3 group-hover:scale-110 transition">
                            {uploading ? <Loader2 className="animate-spin text-blue-400" /> : <Upload className="text-blue-400" />}
                        </div>
                        <h3 className="font-semibold text-gray-200">Upload New PDF</h3>
                        </div>
                    </label>
                </div>

                {/* Saved Documents */}
                {documents.length > 0 && (
                    <div className="max-w-md mx-auto">
                        <h3 className="text-gray-400 text-sm font-bold uppercase tracking-wider mb-3 flex items-center">
                            <History size={16} className="mr-2"/> Recently Uploaded
                        </h3>
                        <div className="space-y-2">
                            {documents.map((doc, i) => (
                                <div key={i} onClick={() => loadOldFile(doc.filename)} 
                                     className="bg-gray-800 p-3 rounded flex justify-between items-center cursor-pointer hover:bg-gray-600 transition border border-gray-700">
                                    <div className="flex items-center">
                                        <FileText size={18} className="text-blue-400 mr-3"/>
                                        <span className="text-sm font-medium">{doc.filename}</span>
                                    </div>
                                    <span className="text-xs text-gray-500 flex items-center">
                                        <Clock size={12} className="mr-1"/> {doc.date.split(' ')[0]}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
          )}
        </div>
      </div>

      {/* RIGHT SIDE: Chat */}
      <div className="w-1/2 flex flex-col bg-gray-900">
        
        {isSpeaking && (
            <div className="absolute top-4 right-4 z-50">
                <button onClick={stopSpeaking} className="bg-red-500 text-white px-4 py-2 rounded-full shadow-lg flex items-center animate-pulse">
                    <StopCircle size={20} className="mr-2"/> Stop Reading
                </button>
            </div>
        )}

        <div className="flex-1 p-6 overflow-y-auto space-y-6">
          {messages.map((msg, index) => (
            <div key={index} className={`flex items-start ${msg.type === 'user' ? 'justify-end' : ''}`}>
              
              {msg.type === 'ai' && (
                  <div className="mr-3 flex flex-col items-center">
                      <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center mb-2"><Bot size={18}/></div>
                      <button onClick={() => speakText(msg.text)} className="text-gray-500 hover:text-blue-400 transition" title="Read Aloud">
                          <Volume2 size={16} />
                      </button>
                  </div>
              )}
              
              <div className={`p-4 rounded-xl max-w-xl ${msg.type === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-200 border border-gray-700'}`}>
                <p className="leading-relaxed whitespace-pre-wrap">{msg.text}</p>
              </div>
              
              {msg.type === 'user' && <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center ml-3"><User size={18}/></div>}
            </div>
          ))}
          {thinking && <div className="flex items-center text-gray-500 text-sm ml-12"><Loader2 className="w-3 h-3 animate-spin mr-2" /> AI is thinking...</div>}
        </div>

        <div className="p-4 border-t border-gray-800 bg-gray-900">
          <div className="flex items-center bg-gray-800 rounded-full px-4 py-3 border border-gray-700 focus-within:border-blue-500 transition shadow-lg">
            <Globe className="text-gray-500 mr-2" size={20} />
            <input 
              type="text" 
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleAsk()}
              placeholder="Ask anything..." 
              className="flex-1 bg-transparent border-none outline-none text-white placeholder-gray-400"
            />
            <button onClick={() => handleAsk()} disabled={!question || thinking} className="p-2 bg-blue-600 hover:bg-blue-500 rounded-full text-white transition disabled:opacity-50 ml-2">
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;