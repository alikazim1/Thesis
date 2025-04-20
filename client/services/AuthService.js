import axios from "axios";

class AuthService {
  async getUser() {
    try {
      const response = await axios.get("http://localhost:3001/user", {
        withCredentials: true,
      });
      return response.data;
    } catch (error) {
      console.error("Failed to fetch user:", error);
      return null;
    }
  }

  async login(credentials) {
    return axios.post("http://localhost:3001/login", credentials, {
      withCredentials: true,
    });
  }

  async signup(data) {
    return axios.post("http://localhost:3001/register", data);
  }

  async logout() {
    return axios.post(
      "http://localhost:3001/logout",
      {},
      {
        withCredentials: true,
      }
    );
  }
}

export default new AuthService();
