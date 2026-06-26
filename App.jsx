import React, { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from "react-router-dom";
import { getToken, clearToken } from "./api/client.js";
import Login from "./pages/Login.jsx";
import Register from "./pages/Register.jsx";
import Shell from "./components/Shell.jsx";
import EtfDashboard from "./pages/EtfDashboard.jsx";
import StocksDashboard from "./pages/StocksDashboard.jsx";
import News from "./pages/News.jsx";

function RequireAuth({ children }) {
  const token = getToken();
  if (!token) return <Navigate to="/login" replace />;
  return children;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route
          path="/*"
          element={
            <RequireAuth>
              <Shell>
                <Routes>
                  <Route path="/" element={<EtfDashboard />} />
                  <Route path="/stocks" element={<StocksDashboard />} />
                  <Route path="/news" element={<News assetType="stock" />} />
                  <Route path="/news/etf" element={<News assetType="etf" />} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </Shell>
            </RequireAuth>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}
