import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Navbar from './Navbar';

export default function Layout() {
  const location = useLocation();
  const isAuthPage = ['/login', '/signup', '/forgot-password'].includes(location.pathname);
  const isChatPage = location.pathname === '/chat';

  if (isChatPage) {
    return <Outlet />;
  }

  return (
    <div className="min-h-screen bg-pattern">
      {!isAuthPage && <Navbar />}
      <main className={`${isAuthPage ? '' : 'container mx-auto px-4 py-8'}`}>
        <Outlet />
      </main>
    </div>
  );
}