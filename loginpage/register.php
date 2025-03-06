<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="stylesheet" href="/style/style.css" />
    <title>Register</title>
  </head>
  <body>
    <div class="container">
      <div class="box form-box">
        <header>Sign Up</header>
        <form action="" method="post">
          <div class="field input">
            <label for="username">Username</label>
            <input
              type="text"
              name="username"
              id="username"
              autocomplete="off"
              required
            />
          </div>

          <div class="field input">
            <label for="email">Email</label>
            <input
              type="text"
              name="email"
              id="email"
              autocomplete="off"
              required
            />
          </div>

          <div class="field input">
            <label for="age">Age</label>
            <input
              type="number"
              name="username"
              id="username"
              autocomplete="off"
              required
            />
          </div>

          <div class="field input">
            <label for="Password">Password</label>
            <input
              type="password"
              name="password"
              id="password"
              autocomplete="off"
              required
            />
          </div>

          <div class="field">
            <input
              type="submit"
              name="submit"
              value="Login"
              autocomplete="off"
              required
            />
          </div>
          <div class="links">
            Already a member? <a href="index.php">Sign In</a>
          </div>
        </form>
      </div>
    </div>
  </body>
</html>
