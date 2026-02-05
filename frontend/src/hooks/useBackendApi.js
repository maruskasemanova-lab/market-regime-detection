import { useState, useEffect, useCallback, useRef } from "react";

const API_URL = "http://localhost:8001";

export function useBackendApi() {
  const [state, setState] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);

  const fetchState = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/state`);
      if (!response.ok) throw new Error("Failed to fetch state");
      const data = await response.json();
      setState(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchRegime = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/regime`);
      if (!response.ok) throw new Error("Failed to fetch regime");
      return await response.json();
    } catch (err) {
      console.error("Regime fetch error:", err);
      return null;
    }
  }, []);

  const fetchSignals = useCallback(async (limit = 50) => {
    try {
      const response = await fetch(`${API_URL}/api/signals?limit=${limit}`);
      if (!response.ok) throw new Error("Failed to fetch signals");
      return await response.json();
    } catch (err) {
      console.error("Signals fetch error:", err);
      return { signals: [], total: 0 };
    }
  }, []);

  const fetchTrades = useCallback(async (limit = 100) => {
    try {
      const response = await fetch(`${API_URL}/api/trades?limit=${limit}`);
      if (!response.ok) throw new Error("Failed to fetch trades");
      return await response.json();
    } catch (err) {
      console.error("Trades fetch error:", err);
      return { trades: [], total: 0 };
    }
  }, []);

  const fetchPositions = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/positions`);
      if (!response.ok) throw new Error("Failed to fetch positions");
      return await response.json();
    } catch (err) {
      console.error("Positions fetch error:", err);
      return { open_positions: {}, count: 0 };
    }
  }, []);

  const fetchPerformance = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/performance`);
      if (!response.ok) throw new Error("Failed to fetch performance");
      return await response.json();
    } catch (err) {
      console.error("Performance fetch error:", err);
      return null;
    }
  }, []);

  const fetchCurrent = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/current`);
      if (!response.ok) throw new Error("Failed to fetch current");
      return await response.json();
    } catch (err) {
      console.error("Current fetch error:", err);
      return null;
    }
  }, []);

  const stepBacktest = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/step`, { method: "POST" });
      if (!response.ok) throw new Error("Failed to step");
      const result = await response.json();
      await fetchState();
      return result;
    } catch (err) {
      console.error("Step error:", err);
      return null;
    }
  }, [fetchState]);

  const runBacktest = useCallback(
    async (bars = 10) => {
      try {
        const response = await fetch(`${API_URL}/api/run?bars=${bars}`, {
          method: "POST",
        });
        if (!response.ok) throw new Error("Failed to run");
        const result = await response.json();
        await fetchState();
        return result;
      } catch (err) {
        console.error("Run error:", err);
        return null;
      }
    },
    [fetchState],
  );

  const resetEngine = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/reset`, { method: "POST" });
      if (!response.ok) throw new Error("Failed to reset");
      const result = await response.json();
      await fetchState();
      return result;
    } catch (err) {
      console.error("Reset error:", err);
      return null;
    }
  }, [fetchState]);

  const toggleStrategy = useCallback(
    async (strategyName, enabled) => {
      try {
        const response = await fetch(`${API_URL}/api/strategies/toggle`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ strategy_name: strategyName, enabled }),
        });
        if (!response.ok) throw new Error("Failed to toggle strategy");
        await fetchState();
        return await response.json();
      } catch (err) {
        console.error("Toggle error:", err);
        return null;
      }
    },
    [fetchState],
  );

  const updateStrategy = useCallback(
    async (strategyName, params) => {
      try {
        const response = await fetch(`${API_URL}/api/strategies/update`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ strategy_name: strategyName, params }),
        });
        if (!response.ok) throw new Error("Failed to update strategy");
        await fetchState();
        return await response.json();
      } catch (err) {
        console.error("Update strategy error:", err);
        return null;
      }
    },
    [fetchState],
  );

  const fetchTradingConfig = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/config/trading`);
      if (!response.ok) throw new Error("Failed to fetch trading config");
      return await response.json();
    } catch (err) {
      console.error("Trading config fetch error:", err);
      return null;
    }
  }, []);

  const updateTradingConfig = useCallback(async (config) => {
    try {
      const response = await fetch(`${API_URL}/api/config/trading`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (!response.ok) throw new Error("Failed to update trading config");
      return await response.json();
    } catch (err) {
      console.error("Update trading config error:", err);
      return null;
    }
  }, []);

  // WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(`ws://localhost:8001/ws`);

    ws.onopen = () => {
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "step_result") {
        fetchState();
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
      // Reconnect after delay
      setTimeout(connectWebSocket, 3000);
    };

    wsRef.current = ws;
  }, [fetchState]);

  // Initial fetch
  useEffect(() => {
    fetchState();
  }, [fetchState]);

  return {
    state,
    loading,
    error,
    fetchState,
    fetchRegime,
    fetchSignals,
    fetchTrades,
    fetchPositions,
    fetchPerformance,
    fetchCurrent,
    stepBacktest,
    runBacktest,
    resetEngine,
    toggleStrategy,
    updateStrategy,
    fetchTradingConfig,
    updateTradingConfig,
    connectWebSocket,
  };
}

export default useBackendApi;
