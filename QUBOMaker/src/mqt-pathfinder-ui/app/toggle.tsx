import React, { useState } from 'react';
import styles from './style.module.css'

interface ToggleProps {
    children: React.ReactNode;
    state?: boolean;
    onUpdate?: (isToggled: boolean) => void;
}

const Toggle: React.FC<ToggleProps> = ({ children, state, onUpdate }) => {
    const [isToggled, setIsToggled] = useState(false);

    const getIsToggled = () => {
        return state ?? isToggled;
    }

    const handleToggle = () => {
        const newState = !getIsToggled();
        setIsToggled(newState);
        onUpdate?.(newState);
    };

    return (
        <button onClick={handleToggle} className={`rounded border-2 p-1 text-center cursor-pointer bg-${getIsToggled() ? "slate-100" : "white"}`}>
            {children}
        </button>
    );
};

export default Toggle;
