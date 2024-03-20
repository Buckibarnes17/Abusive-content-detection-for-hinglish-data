import React from 'react'
import SignUp from './pages/Signup/SignUp'
import Login from './pages/Login/Login'
import Home from './pages/Home/Home'
import {BrowserRouter, Route, Routes} from 'react-router-dom'

function App() {
  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route path = '/' element={<Home/>}></Route>
          <Route path = '/login' element={<Login/>}></Route>
          <Route path = '/signup' element={<SignUp/>}></Route>
        </Routes>
      </BrowserRouter>
    </>
  )
}

export default App
