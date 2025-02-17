const signUpButton = document.getElementById('signUp');
const signInButton = document.getElementById('signIn');
const container = document.getElementById('container');

signUpButton.addEventListener('click', () => {
  container.classList.add('right-panel-active');
});

signInButton.addEventListener('click', () => {
  container.classList.remove('right-panel-active');
});

// Signup form submission
document.getElementById('signupForm').addEventListener('submit', (e) => {
  e.preventDefault();
  const name = document.getElementById('signupName').value;
  const email = document.getElementById('signupEmail').value;
  const password = document.getElementById('signupPassword').value;

  if (name && email && password) {
    // Save the data in localStorage
    const userData = { name, email, password };
    localStorage.setItem('userData', JSON.stringify(userData));
    alert('Signup successful!');
  } else {
    alert('Please fill in all fields.');
  }
});

// Login form submission
document.getElementById('loginForm').addEventListener('submit', (e) => {
  e.preventDefault();
  const email = document.getElementById('loginEmail').value;
  const password = document.getElementById('loginPassword').value;

  const storedData = localStorage.getItem('userData');
  if (storedData) {
    const userData = JSON.parse(storedData);
    if (userData.email === email && userData.password === password) {
      alert('Login successful!');
      // Redirect to home.html after the user clicks "OK" in the alert
      window.location.href = 'index1.html';
    } else {
      alert('Incorrect email or password.');
    }
  } else {
    alert('No user data found. Please sign up first.');
  }
});
