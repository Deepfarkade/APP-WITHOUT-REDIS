import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { KeyRound, Mail } from 'lucide-react';
import { AuthService } from '../services/authService';

export default function ForgotPasswordForm() {
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage('');
    
    try {
      await AuthService.forgotPassword(email);
      setMessage('Password reset link has been sent to your email');
      setTimeout(() => navigate('/login'), 3000);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to send reset link');
    } finally {
      setIsLoading(false);
    }
  };

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
            Reset Password
          </h2>

          {message && (
            <div className={`p-4 rounded-md mb-6 ${
              message.includes('sent') 
                ? 'bg-green-900/20 text-green-200'
                : 'bg-red-900/20 text-red-200'
            }`}>
              {message}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Email Address
              </label>
              <div className="relative">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-4 py-3 bg-secondary border border-secondary-light rounded-lg text-white focus:ring-2 focus:ring-primary focus:border-transparent"
                  placeholder="Enter your email"
                  required
                />
                <Mail className="absolute right-3 top-3 h-5 w-5 text-gray-400" />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-primary hover:bg-primary-dark text-secondary-dark font-semibold py-3 px-4 rounded-lg transition duration-200 flex items-center justify-center space-x-2"
            >
              <KeyRound className="h-5 w-5" />
              <span>{isLoading ? 'Sending...' : 'Reset Password'}</span>
            </button>

            <div className="text-center">
              <button
                type="button"
                onClick={() => navigate('/login')}
                className="text-primary hover:text-primary-light text-sm"
              >
                Back to Login
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}