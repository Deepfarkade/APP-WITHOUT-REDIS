import React from 'react';
import { Link } from 'react-router-dom';
import SignupForm from '../features/auth/components/SignupForm';

export default function SignupPage() {
  return (
    <div className="min-h-screen bg-pattern">
      <div className="container mx-auto px-4 flex flex-col items-center justify-center min-h-screen">
        <div className="w-full max-w-md">
          <h1 className="text-4xl font-bold text-white text-center mb-8">
            Create your account
          </h1>
          <SignupForm />
          <p className="mt-4 text-center text-gray-400">
            Already have an account?{' '}
            <Link to="/login" className="text-primary hover:text-primary-light font-medium">
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}