/**
 * useNotifications Hook
 * Notification state management with WebSocket integration and auto-fetch
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { DefectAlert } from './useWebSocketStream';

const API_BASE_URL = 'http://localhost:8000';

export interface Notification {
  id: string;
  type: 'defect' | 'parameter_anomaly' | 'prediction' | 'system' | 'recommendation';
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  message: string;
  timestamp: string;
  acknowledged: boolean;
  details?: any;
}

export interface UseNotificationsReturn {
  notifications: Notification[];
  unacknowledgedCount: number;
  addNotification: (notification: Omit<Notification, 'acknowledged'>) => void;
  acknowledgeNotification: (id: string) => void;
  dismissNotification: (id: string) => void;
  clearAll: () => void;
  fetchNotifications: () => Promise<void>;
}

export const useNotifications = (defectAlerts?: DefectAlert[]): UseNotificationsReturn => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const processedAlertsRef = useRef<Set<string>>(new Set());

  // Initialize notification sound (optional)
  useEffect(() => {
    audioRef.current = new Audio('/notification.mp3');
    audioRef.current.volume = 0.5;
  }, []);

  const playNotificationSound = useCallback((severity: string) => {
    // Only play sound for HIGH and CRITICAL notifications
    if ((severity === 'HIGH' || severity === 'CRITICAL') && audioRef.current) {
      audioRef.current.play().catch(err => {
        console.warn('Could not play notification sound:', err);
      });
    }
  }, []);

  const addNotification = useCallback((notification: Omit<Notification, 'acknowledged'>) => {
    const newNotification: Notification = {
      ...notification,
      acknowledged: false
    };

    setNotifications(prev => {
      // Prevent duplicates
      if (prev.some(n => n.id === newNotification.id)) {
        return prev;
      }
      
      // Keep only last 100 notifications
      const updated = [newNotification, ...prev].slice(0, 100);
      
      // Play sound for critical notifications
      playNotificationSound(newNotification.severity);
      
      return updated;
    });
  }, [playNotificationSound]);

  const acknowledgeNotification = useCallback((id: string) => {
    setNotifications(prev =>
      prev.map(n => n.id === id ? { ...n, acknowledged: true } : n)
    );
  }, []);

  const dismissNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  const clearAll = useCallback(() => {
    setNotifications([]);
    processedAlertsRef.current.clear();
  }, []);

  const fetchNotifications = useCallback(async () => {
    try {
      // Fetch alerts from backend
      const alertsResponse = await fetch(`${API_BASE_URL}/api/alerts/active`);
      if (alertsResponse.ok) {
        const alerts = await alertsResponse.json();
        
        alerts.forEach((alert: any) => {
          if (!processedAlertsRef.current.has(alert.alert_id)) {
            addNotification({
              id: alert.alert_id,
              type: 'system',
              severity: alert.priority as any,
              message: alert.message,
              timestamp: alert.timestamp,
              details: alert
            });
            processedAlertsRef.current.add(alert.alert_id);
          }
        });
      }

      // Fetch recommendations
      const recsResponse = await fetch(`${API_BASE_URL}/api/recommendations`);
      if (recsResponse.ok) {
        const recs = await recsResponse.json();
        
        recs.forEach((rec: any) => {
          const recId = rec.recommendation_id;
          if (!processedAlertsRef.current.has(recId)) {
            addNotification({
              id: recId,
              type: 'recommendation',
              severity: rec.urgency === 'HIGH' ? 'HIGH' : 'MEDIUM',
              message: rec.description,
              timestamp: rec.timestamp,
              details: rec
            });
            processedAlertsRef.current.add(recId);
          }
        });
      }
    } catch (error) {
      console.error('Error fetching notifications:', error);
    }
  }, [addNotification]);

  // Auto-fetch notifications on mount and periodically
  useEffect(() => {
    fetchNotifications();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchNotifications, 30000);
    
    return () => clearInterval(interval);
  }, [fetchNotifications]);

  // Process defect alerts from WebSocket
  useEffect(() => {
    if (!defectAlerts || defectAlerts.length === 0) return;

    defectAlerts.forEach(alert => {
      const alertId = `defect_${alert.timestamp}`;
      
      // Skip if already processed
      if (processedAlertsRef.current.has(alertId)) {
        return;
      }

      // Check for parameter anomalies
      const tempDeviation = Math.abs(alert.parameters_snapshot.furnace_temperature - 1520);
      const speedDeviation = Math.abs(alert.parameters_snapshot.belt_speed - 150);
      
      // Add defect notification
      addNotification({
        id: alertId,
        type: 'defect',
        severity: alert.severity,
        message: `${alert.defect_type.toUpperCase()} detected at Line ${alert.location.line} (confidence: ${(alert.confidence * 100).toFixed(0)}%)`,
        timestamp: alert.timestamp,
        details: alert
      });

      // Add parameter anomaly notification if significant deviation
      if (tempDeviation > 40 || speedDeviation > 30) {
        const anomalyId = `anomaly_${alert.timestamp}`;
        addNotification({
          id: anomalyId,
          type: 'parameter_anomaly',
          severity: tempDeviation > 60 || speedDeviation > 50 ? 'HIGH' : 'MEDIUM',
          message: `Parameter anomaly detected: ${
            tempDeviation > 40 
              ? `Temperature ${tempDeviation.toFixed(0)}°C from target` 
              : `Speed ${speedDeviation.toFixed(0)} m/min from target`
          }`,
          timestamp: alert.timestamp,
          details: {
            temperature_deviation: tempDeviation,
            speed_deviation: speedDeviation
          }
        });
        processedAlertsRef.current.add(anomalyId);
      }

      processedAlertsRef.current.add(alertId);
    });
  }, [defectAlerts, addNotification]);

  const unacknowledgedCount = notifications.filter(n => !n.acknowledged).length;

  return {
    notifications,
    unacknowledgedCount,
    addNotification,
    acknowledgeNotification,
    dismissNotification,
    clearAll,
    fetchNotifications
  };
};
