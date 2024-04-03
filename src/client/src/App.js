import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './components/Home';
import StationInfo from './components/StationInfo';
import Header from './components/Header';

const routes = [
  { path: '/', element: <Home />, private: true },
  { path: '/station/:stationId', element: <StationInfo /> },
];

function App() {
  return (
    <Router>
      <Header />
      <Routes>
        {routes.map((route, index) => (
          <Route key={index} path={route.path} element={route.element} />
        ))}
      </Routes>
    </Router>
  );
}

export default App;
