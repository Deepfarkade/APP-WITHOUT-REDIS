import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { UserPlus, Mail, Lock, User, ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function SignupForm() {
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const { register } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords don't match");
      return;
    }

    if (formData.password.length < 8) {
      setError("Password must be at least 8 characters long");
      return;
    }

    try {
      setIsLoading(true);
      await register(formData.email, formData.password, formData.fullName);
      navigate('/login', { 
        state: { message: 'Registration successful! Please login.' }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-pattern flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <div className="flex justify-center">
          <div className="flex items-center relative">
            <span className="font-bold text-4xl text-primary animate-pulse">EY</span>
            <div className="w-6 h-6 bg-primary ml-1 transform -skew-x-12 animate-pulse"></div>
            <div className="absolute -inset-4 bg-primary opacity-20 blur-lg rounded-full"></div>
          </div>
        </div>
        <h2 className="mt-6 text-center text-3xl font-extrabold text-white">
          <span className="gradient-text">Create Your Account</span>
        </h2>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="auth-container py-8 px-4 shadow sm:rounded-lg sm:px-10">
          {error && (
            <div className="mb-4 p-3 text-sm text-red-200 bg-red-900/50 rounded-md backdrop-blur-sm animate-fadeIn">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="fullName" className="block text-sm font-medium text-white">
                Full Name
              </label>
              <div className="mt-1 relative">
                <input
                  type="text"
                  id="fullName"
                  value={formData.fullName}
                  onChange={(e) => setFormData({ ...formData, fullName: e.target.value })}
                  className="form-input block w-full rounded-md bg-secondary border-secondary-light text-white shadow-sm focus:border-primary focus:ring-primary transition-all duration-300"
                  required
                  disabled={isLoading}
                />
                <User className="absolute right-3 top-2.5 h-5 w-5 text-gray-400" />
              </div>
            </div>

            <div>
              <label htmlFor="email" className="block text-sm font-medium text-white">
                Email
              </label>
              <div className="mt-1 relative">
                <input
                  type="email"
                  id="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  className="form-input block w-full rounded-md bg-secondary border-secondary-light text-white shadow-sm focus:border-primary focus:ring-primary transition-all duration-300"
                  required
                  disabled={isLoading}
                />
                <Mail className="absolute right-3 top-2.5 h-5 w-5 text-gray-400" />
              </div>
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-white">
                Password
              </label>
              <div className="mt-1 relative">
                <input
                  type="password"
                  id="password"
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  className="form-input block w-full rounded-md bg-secondary border-secondary-light text-white shadow-sm focus:border-primary focus:ring-primary transition-all duration-300"
                  required
                  minLength={8}
                  disabled={isLoading}
                />
                <Lock className="absolute right-3 top-2.5 h-5 w-5 text-gray-400" />
              </div>
            </div>

            <div>
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-white">
                Confirm Password
              </label>
              <div className="mt-1 relative">
                <input
                  type="password"
                  id="confirmPassword"
                  value={formData.confirmPassword}
                  onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                  className="form-input block w-full rounded-md bg-secondary border-secondary-light text-white shadow-sm focus:border-primary focus:ring-primary transition-all duration-300"
                  required
                  minLength={8}
                  disabled={isLoading}
                />
                <Lock className="absolute right-3 top-2.5 h-5 w-5 text-gray-400" />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="hover-effect w-full flex justify-center items-center py-2 px-4 border border-transparent rounded-md shadow-sm text-secondary-dark bg-primary hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary disabled:opacity-50 font-medium transition-all duration-300"
            >
              {isLoading ? (
                <div className="loading-spinner"></div>
              ) : (
                <>
                  <UserPlus className="w-5 h-5 mr-2" />
                  <span>Create Account</span>
                </>
              )}
            </button>
          </form>

          <div className="mt-6 text-center">
            <Link
              to="/login"
              className="group inline-flex items-center text-primary hover:text-primary-light font-medium transition-all duration-300"
            >
              <ArrowLeft className="mr-2 w-4 h-4 transform group-hover:-translate-x-1 transition-transform duration-300" />
              Back to Login
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}