import React, { useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { AuthService } from '../features/auth/services/authService';
import { KeyRound } from 'lucide-react';

export default function ResetPasswordPage() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token');
  const navigate = useNavigate();
  
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!token) {
      setError('Invalid reset token');
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    try {
      setIsLoading(true);
      setError('');
      await AuthService.resetPassword(token, newPassword);
      setSuccess(true);
      setTimeout(() => navigate('/login'), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reset password');
    } finally {
      setIsLoading(false);
    }
  };

  if (!token) {
    return (
      <div className="min-h-screen bg-secondary flex items-center justify-center px-4">
        <div className="bg-secondary-dark p-8 rounded-lg shadow-xl">
          <p className="text-red-400">Invalid or missing reset token</p>
          <button
            onClick={() => navigate('/forgot-password')}
            className="mt-4 text-primary hover:text-primary-light"
          >
            Request new reset link
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-secondary flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="bg-secondary-dark p-8 rounded-lg shadow-xl">
          <div className="flex justify-center mb-8">
            <div className="flex items-center">
              <span className="font-bold text-4xl text-primary">EY</span>
              <div className="w-6 h-6 bg-primary ml-1 transform -skew-x-12"></div>
            </div>
          </div>

          <h2 className="text-2xl font-bold text-white text-center mb-6">
            Reset Your Password
          </h2>

          {error && (
            <div className="bg-red-900/20 text-red-200 p-4 rounded-md mb-6">
              {error}
            </div>
          )}

          {success && (
            <div className="bg-green-900/20 text-green-200 p-4 rounded-md mb-6">
              Password reset successful! Redirecting to login...
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-white mb-2">
                New Password
              </label>
              <input
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                className="w-full px-4 py-3 bg-secondary border border-secondary-light rounded-lg text-white focus:ring-2 focus:ring-primary focus:border-transparent"
                required
                minLength={8}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Confirm New Password
              </label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full px-4 py-3 bg-secondary border border-secondary-light rounded-lg text-white focus:ring-2 focus:ring-primary focus:border-transparent"
                required
                minLength={8}
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-primary hover:bg-primary-dark text-secondary-dark font-semibold py-3 px-4 rounded-lg transition duration-200 flex items-center justify-center space-x-2"
            >
              <KeyRound className="h-5 w-5" />
              <span>{isLoading ? 'Resetting...' : 'Reset Password'}</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}