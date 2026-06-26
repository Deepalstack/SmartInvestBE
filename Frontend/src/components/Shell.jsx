import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { clearToken, getUsername } from "../api/client.js";

export default function Shell({ children }) {
  const navigate = useNavigate();
  const username = getUsername();

  function handleLogout() {
    clearToken();
    navigate("/login");
  }

  return (
    <div className="app-shell">
      <nav className="rail">
        <div className="rail-brand">
          smart<span className="dot">·</span>invest
        </div>
        <div className="rail-nav">
          <NavLink to="/" className={({ isActive }) => `rail-link${isActive ? " active" : ""}`}>
            ETF Ledger
          </NavLink>
          <NavLink to="/stocks" className={({ isActive }) => `rail-link${isActive ? " active" : ""}`}>
            Stock Ledger
          </NavLink>
          <NavLink to="/news" className={({ isActive }) => `rail-link${isActive ? " active" : ""}`}>
            News — Stocks
          </NavLink>
          <NavLink to="/news/etf" className={({ isActive }) => `rail-link${isActive ? " active" : ""}`}>
            News — ETFs
          </NavLink>
        </div>
        <div className="rail-footer">
          <div>{username}</div>
          <a className="rail-logout" onClick={handleLogout} role="button" tabIndex={0}>
            Sign out
          </a>
        </div>
      </nav>
      <main className="main">{children}</main>
    </div>
  );
}
