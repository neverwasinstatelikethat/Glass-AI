import React from 'react';
import ReactDOM from 'react-dom/client';
import AppWithMaterialUI from './AppWithMaterialUI';

console.log('index.tsx loaded');
console.log('AppWithMaterialUI imported');

const container = document.getElementById('root');
console.log('Root element found:', container);

if (container) {
  const root = ReactDOM.createRoot(container);
  console.log('Root created');
  
  root.render(
    <React.StrictMode>
      <AppWithMaterialUI />
    </React.StrictMode>
  );
  
  console.log('App rendered');
} else {
  console.error('Root element not found!');
}