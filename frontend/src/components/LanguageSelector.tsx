import React from 'react';

interface LanguageOption {
  code: string;
  name: string;
}

const SUPPORTED_LANGUAGES: LanguageOption[] = [
  { code: "eng", name: "English" },
  { code: "spa", name: "Spanish" },
  { code: "fra", name: "French" },
  { code: "deu", name: "German" },
  { code: "cmn", name: "Chinese" },
  { code: "jpn", name: "Japanese" },
  { code: "kor", name: "Korean" },
  { code: "ara", name: "Arabic" },
];

interface LanguageSelectorProps {
  onLanguageChange: (language: string) => void;
  currentLanguage: string;
}

export const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  onLanguageChange,
  currentLanguage
}) => {
  return (
    <div className="language-selector">
      <select
        value={currentLanguage}
        onChange={(e) => onLanguageChange(e.target.value)}
      >
        {SUPPORTED_LANGUAGES.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.name}
          </option>
        ))}
      </select>
    </div>
  );
}; 