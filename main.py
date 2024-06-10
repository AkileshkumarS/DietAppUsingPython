import sqlite3
from sqlite3 import Error
import streamlit as st
from app1 import app as app1
from app2 import app as app2
from app4 import app as app4

# Function to create a SQLite database connection
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn

# Function to create a new user in the database
def create_user(conn, user):
    sql = ''' INSERT INTO users(username,password,email)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, user)
    conn.commit()
    return cur.lastrowid

# Function to retrieve user details by username
def get_user_by_username(conn, username):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    rows = cur.fetchall()
    return rows


# Function to validate email format
def is_valid_email(email):
    import re
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# Function to view all users in the database
def view_all_users(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users")
    rows = cur.fetchall()
    return rows

# Function to create the login page
def login_page(conn):
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = get_user_by_username(conn, username)
        if len(user) == 0:
            st.error("User not found")
        else:
            if user[0][2] == password:  # Check if the password matches
                st.success("Login successful!")
                st.session_state.logged_in = True
                st.session_state.username = username  # Store username in session state
            else:
                st.error("Incorrect password")

def sign_up_page(conn):
    st.title("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    email = st.text_input("Email")

    if st.button("Sign Up"):
        if len(password) < 8:
            st.error("Password must be at least 8 characters long")
        elif not is_valid_email(email):
            st.error("Invalid email format")
        else:
            user = get_user_by_username(conn, username)
            if len(user) > 0:
                st.error("Username already exists")
            else:
                create_user(conn, (username, password, email))
                st.success("Sign up successful! Please login.")

# Function to create the forgot password page
def forgot_password_page(conn):
    st.title("Forgot Password")
    username = st.text_input("Username")
    new_password = st.text_input("New Password", type="password")

    if st.button("Reset Password"):
        user = get_user_by_username(conn, username)
        if len(user) == 0:
            st.error("User not found")
        else:
          if len(new_password) < 8:
            st.error("New Password must be at least 8 characters long")
          else:
            # Update the password in the database
            cur = conn.cursor()
            cur.execute("UPDATE users SET password=? WHERE username=?", (new_password, username))
            conn.commit()
            st.success("Password reset successful! Please login with your new password.")

def home(conn):

  if st.session_state.username == "admin" and st.session_state.logged_in:
    st.sidebar.title("Database")
    if st.sidebar.button("View"):
      st.subheader("Database View")
      users = view_all_users(conn)
      if len(users) > 0:
        st.write("All Users:")
        for user in users:
          st.write(user)
      else:
        st.write("No users found")
        
# Sidebar with styled radio buttons for page navigation
  st.sidebar.title('Page Navigation')
  selection = st.sidebar.radio("Go to",['Home Page', 'PCOS Diet Recommendation Page', 'PCOS Ultrasound Image Classification Page'])
  
  if selection == 'Home Page':
    app4()
  elif selection == 'PCOS Diet Recommendation Page':
    app1()
  elif selection == 'PCOS Ultrasound Image Classification Page':
    app2()


  st.sidebar.title("Logout")
  if st.sidebar.button("Click here to Logout"):
    st.session_state.logged_in = False

# Main function
def main():
    # Create or connect to the SQLite database
    conn = create_connection("user_database.db")

    if not conn:
        st.error("Error creating database connection")

    # Create a table for users if it doesn't exist
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users 
                   (id INTEGER PRIMARY KEY, username TEXT, password TEXT, email TEXT)''')

    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Display appropriate page based on session state
    if st.session_state.logged_in:
        home(conn)
    else:
        st.markdown("---")
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Login", "Sign Up", "Forgot Password"])

        if page == "Login":
            login_page(conn)
        elif page == "Sign Up":
            sign_up_page(conn)
        elif page == "Forgot Password":
            forgot_password_page(conn)

    conn.close()

if __name__ == "__main__":
    main()