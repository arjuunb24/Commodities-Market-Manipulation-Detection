import ReactECharts from "echarts-for-react";
import { connect, type EChartsOption } from "echarts";
import { memo } from "react";

interface Props {
  option: EChartsOption;
  height?: number | string;
  className?: string;
  onEvents?: Record<string, (params: unknown) => void>;
  /** Charts in the same group share crosshair and zoom. */
  group?: string;
}

/** Thin wrapper: canvas renderer, merges updates so data changes animate instead of re-mounting. */
function EChartBase({ option, height = 300, className, onEvents, group }: Props) {
  return (
    <ReactECharts
      option={option}
      style={{ height, width: "100%" }}
      className={className}
      opts={{ renderer: "canvas" }}
      notMerge
      lazyUpdate
      onEvents={onEvents}
      onChartReady={(chart) => {
        if (group) {
          chart.group = group;
          connect(group);
        }
      }}
    />
  );
}

export const EChart = memo(EChartBase);
