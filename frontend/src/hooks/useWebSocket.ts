import { useState, useEffect, useRef } from 'react';

interface WebSocketMessage {
  type: string;
  data?: any;
  [key: string]: any;
}

interface UseWebSocketOptions {
  onMessage?: (data: WebSocketMessage) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
}

const useWebSocket = (url: string, options: UseWebSocketOptions = {}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Event | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Create WebSocket connection
    const ws = new WebSocket(url);
    wsRef.current = ws;

    // Connection opened
    ws.onopen = () => {
      console.log('WebSocket connection established');
      setIsConnected(true);
      setError(null);
      if (options.onOpen) {
        options.onOpen();
      }
    };

    // Listen for messages
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('WebSocket message received:', data);
        
        // Handle different message types
        if (data.type === 'pipeline_result') {
          console.log('Pipeline result received:', data.data);
        } else if (data.type === 'defect_alert') {
          console.log('Defect alert received:', data.data);
        } else if (data.type === 'recommendation') {
          console.log('Recommendation received:', data.data);
        }
        
        if (options.onMessage) {
          options.onMessage(data);
        }
      } catch (e) {
        console.error('Error parsing WebSocket message:', event.data);
      }
    };

    // Connection closed
    ws.onclose = () => {
      console.log('WebSocket connection closed');
      setIsConnected(false);
      if (options.onClose) {
        options.onClose();
      }
    };

    // Connection error
    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      setError(err);
      setIsConnected(false);
      if (options.onError) {
        options.onError(err);
      }
    };

    // Cleanup function
    return () => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    };
  }, [url, options]);

  const sendMessage = (message: WebSocketMessage) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected. Message not sent:', message);
    }
  };

  const sendPing = () => {
    sendMessage({ type: 'ping' });
  };

  const subscribe = (topics: string[]) => {
    sendMessage({ type: 'subscribe', topics });
  };

  return {
    isConnected,
    error,
    sendMessage,
    sendPing,
    subscribe
  };
};

export default useWebSocket;