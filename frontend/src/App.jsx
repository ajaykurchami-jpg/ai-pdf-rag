import React, { useState } from 'react';
import axios from 'axios';
import { Send, Upload, Loader2, Bot, User, FileText, Trash2, Globe, FileOutput } from 'lucide-react';

// --- CONFIGURATION ---
// ⚠️ IMPORTANT: Verify this is your exact Railway URL. 
// It MUST start with "https://" and MUST NOT have a slash "/" at the end.
const API_BASE_URL = "https://ai-pdf-rag-production.up.railway.app"; 

function App() {
  const [file, setFile] = useState(null);
  const [pdfUrl, setPdfUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([
    { type: 'ai', text: "Hello! Upload a PDF to start. I can answer in any language!" }
  ]);
  const [thinking, setThinking] = useState(false);

  // 1. Reset Chat
  const handleReset = () => {
    setFile(null);
    setPdfUrl(null);
    setMessages([{ type: 'ai', text: "Ready for a new document! Upload one to begin." }]);
    setQuestion("");
  };

  // 2. Upload Logic
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
      // FIX: Use the secure HTTPS variable defined at the top
      await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      
      // FIX: Use the secure HTTPS variable for the PDF viewer too
      setPdfUrl(`${API_BASE_URL}/static/${selectedFile.name}?t=${Date.now()}`);
      
      setMessages(prev => [...prev, { type: 'ai', text: `I've read ${selectedFile.name}. Ask me anything!` }]);
    } catch (error) {
      console.error("Upload Error:", error);
      setMessages(prev => [...prev, { type: 'ai', text: "Error uploading file. Check Backend connection." }]);
    } finally {
      setUploading(false);
    }
  };

  // 3. Ask Question Logic
  const handleAsk = async (customQuestion = null) => {
    const textToSend = customQuestion || question;
    if (!textToSend.trim()) return;

    if (!customQuestion) {
        setMessages(prev => [...prev, { type: 'user', text: textToSend }]);
    } else {
        setMessages(prev => [...prev, { type: 'user', text: "✨ Generating Document Summary..." }]);
    }
    
    setQuestion("");
    setThinking(true);

    try {
      // FIX: Use the secure HTTPS variable
      const res = await axios.post(`${API_BASE_URL}/query`, { question: textToSend });
      setMessages(prev => [...prev, { type: 'ai', text: res.data.answer }]);
    } catch (error) {
      console.error("Chat Error:", error);
      setMessages(prev => [...prev, { type: 'ai', text: "Error: Could not get answer." }]);
    } finally {
      setThinking(false);
    }
  };

  // 4. Auto-Summary Button Handler
  const handleSummary = () => {
    handleAsk("Generate a structured summary of this document in 5 key bullet points. Keep it concise.");
  };

  return (
    <div className="flex h-screen bg-gray-900 text-white font-sans overflow-hidden">
      
      {/* LEFT SIDE: PDF Viewer */}
      <div className="w-1/2 border-r border-gray-700 bg-gray-800 flex flex-col">
        
        {/* Header */}
        <div className="p-4 border-b border-gray-700 flex justify-between items-center bg-gray-900">
          <h1 className="text-xl font-bold text-blue-400 flex items-center">
            <FileText className="mr-2" /> Document Viewer
          </h1>
          
          {/* Action Buttons */}
          {pdfUrl && (
            <div className="flex space-x-2">
                <button onClick={handleSummary} className="bg-green-600 hover:bg-green-500 text-white px-3 py-1 rounded text-sm flex items-center transition">
                    <FileOutput size={16} className="mr-1"/> Summary
                </button>
                <button onClick={handleReset} className="bg-red-600 hover:bg-red-500 text-white px-3 py-1 rounded text-sm flex items-center transition">
                    <Trash2 size={16} className="mr-1"/> Reset
                </button>
            </div>
          )}
        </div>

        {/* PDF Content */}
        <div className="flex-1 bg-gray-700 relative">
          {pdfUrl ? (
            <iframe src={pdfUrl} className="w-full h-full border-none" title="PDF Viewer" />
          ) : (
            <div className="flex flex-col items-center justify-center h-full p-10 text-center">
              <label className="border-2 border-dashed border-gray-500 rounded-xl p-10 hover:border-blue-500 hover:bg-gray-700/50 transition cursor-pointer group">
                <input type="file" onChange={handleFileChange} className="hidden" accept=".pdf" />
                <div className="flex flex-col items-center">
                  <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition">
                    {uploading ? <Loader2 className="animate-spin text-blue-400" /> : <Upload className="text-blue-400" />}
                  </div>
                  <h3 className="text-lg font-semibold text-gray-200">Upload PDF Document</h3>
                  <p className="text-gray-400 text-sm mt-2">Supports Multilingual Q&A + Summarization</p>
                </div>
              </label>
            </div>
          )}
        </div>
      </div>

      {/* RIGHT SIDE: Chat */}
      <div className="w-1/2 flex flex-col bg-gray-900">
        <div className="flex-1 p-6 overflow-y-auto space-y-6">
          {messages.map((msg, index) => (
            <div key={index} className={`flex items-start ${msg.type === 'user' ? 'justify-end' : ''}`}>
              {msg.type === 'ai' && <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center mr-3"><Bot size={18}/></div>}
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