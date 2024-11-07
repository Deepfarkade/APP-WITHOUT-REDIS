import { api } from '../../../lib/api';

interface LoginResponse {
  access_token: string;
  token_type: string;
  user: {
    id: string;
    email: string;
    full_name: string;
  };
}

export class AuthService {
  private static readonly TOKEN_KEY = 'auth_token';
  private static readonly USER_KEY = 'user';
  private static readonly SESSION_DURATION = 30 * 24 * 60 * 60 * 1000; // 30 days

  static async login(email: string, password: string): Promise<LoginResponse> {
    try {
      const formData = new URLSearchParams();
      formData.append('username', email);
      formData.append('password', password);

      const response = await api.post<LoginResponse>('/auth/login', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });

      if (response.data.access_token) {
        localStorage.setItem(this.TOKEN_KEY, response.data.access_token);
        localStorage.setItem(this.USER_KEY, JSON.stringify(response.data.user));
        localStorage.setItem('session_expiry', (Date.now() + this.SESSION_DURATION).toString());
      }

      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Login failed');
    }
  }

  static async register(email: string, password: string, fullName: string): Promise<void> {
    try {
      await api.post('/auth/register', {
        email,
        password,
        full_name: fullName
      });
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Registration failed');
    }
  }

  static async logout(): Promise<void> {
    try {
      await api.post('/auth/logout');
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      this.clearAuthData();
    }
  }

  static async forgotPassword(email: string): Promise<void> {
    try {
      await api.post('/auth/forgot-password', { email });
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to process password reset');
    }
  }

  static async resetPassword(token: string, newPassword: string): Promise<void> {
    try {
      await api.post('/auth/reset-password', { token, new_password: newPassword });
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to reset password');
    }
  }

  static clearAuthData(): void {
    localStorage.removeItem(this.TOKEN_KEY);
    localStorage.removeItem(this.USER_KEY);
    localStorage.removeItem('session_expiry');
  }

  static async initializeAuth(): Promise<LoginResponse | null> {
    try {
      const token = localStorage.getItem(this.TOKEN_KEY);
      const userStr = localStorage.getItem(this.USER_KEY);
      const sessionExpiry = localStorage.getItem('session_expiry');

      if (!token || !userStr || !sessionExpiry) {
        return null;
      }

      if (Date.now() > parseInt(sessionExpiry)) {
        this.clearAuthData();
        return null;
      }

      const user = JSON.parse(userStr);
      return {
        access_token: token,
        token_type: 'bearer',
        user
      };
    } catch (error) {
      console.error('Auth initialization error:', error);
      this.clearAuthData();
      return null;
    }
  }
}