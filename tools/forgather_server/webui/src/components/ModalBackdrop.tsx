import {
  HTMLAttributes,
  MouseEvent,
  ReactNode,
  useRef,
} from "react";

interface ModalBackdropProps
  extends Omit<HTMLAttributes<HTMLDivElement>, "onMouseDown" | "onMouseUp"> {
  onClose: () => void;
  children: ReactNode;
}

// Wraps modal content with a click-to-dismiss backdrop. Only closes when both
// mousedown and mouseup target the backdrop itself; this prevents an unwanted
// close when the user selects text inside an input and releases the mouse
// outside the modal (which would otherwise fire a click on the backdrop).
export function ModalBackdrop({
  onClose,
  children,
  className = "modal-backdrop",
  ...rest
}: ModalBackdropProps) {
  const downOnBackdrop = useRef(false);

  const handleMouseDown = (e: MouseEvent<HTMLDivElement>) => {
    downOnBackdrop.current = e.target === e.currentTarget;
  };

  const handleMouseUp = (e: MouseEvent<HTMLDivElement>) => {
    const close =
      downOnBackdrop.current && e.target === e.currentTarget;
    downOnBackdrop.current = false;
    if (close) onClose();
  };

  return (
    <div
      className={className}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      {...rest}
    >
      {children}
    </div>
  );
}
