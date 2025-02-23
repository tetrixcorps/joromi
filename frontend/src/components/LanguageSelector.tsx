import React from 'react';

interface LanguageOption {
  code: string;
  name: string;
  isAfrican?: boolean;
}

const SUPPORTED_LANGUAGES: LanguageOption[] = [
  // African Languages
  { code: "afr", name: "Afrikaans", isAfrican: true },
  { code: "amh", name: "Amharic (አማርኛ)", isAfrican: true },
  { code: "hau", name: "Hausa (Hausa)", isAfrican: true },
  { code: "ibo", name: "Igbo (Igbo)", isAfrican: true },
  { code: "lin", name: "Lingala (Lingála)", isAfrican: true },
  { code: "mlg", name: "Malagasy", isAfrican: true },
  { code: "nya", name: "Nyanja/Chichewa (Chichewa)", isAfrican: true },
  { code: "orm", name: "Oromo (Oromoo)", isAfrican: true },
  { code: "sna", name: "Shona (chiShona)", isAfrican: true },
  { code: "som", name: "Somali (Soomaali)", isAfrican: true },
  { code: "swh", name: "Swahili (Kiswahili)", isAfrican: true },
  { code: "wol", name: "Wolof (Wolof)", isAfrican: true },
  { code: "xho", name: "Xhosa (isiXhosa)", isAfrican: true },
  { code: "yor", name: "Yoruba (Yorùbá)", isAfrican: true },
  { code: "zul", name: "Zulu (isiZulu)", isAfrican: true },
  
  // Other Languages
  { code: "eng", name: "English" },
  { code: "spa", name: "Spanish (Español)" },
  { code: "fra", name: "French (Français)" },
  { code: "deu", name: "German (Deutsch)" },
  { code: "cmn", name: "Chinese (中文)" },
  { code: "jpn", name: "Japanese (日本語)" },
  { code: "kor", name: "Korean (한국어)" },
  { code: "ara", name: "Arabic (العربية)" },
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
        <optgroup label="African Languages">
          {SUPPORTED_LANGUAGES
            .filter(lang => lang.isAfrican)
            .map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.name}
              </option>
            ))}
        </optgroup>
        <optgroup label="Other Languages">
          {SUPPORTED_LANGUAGES
            .filter(lang => !lang.isAfrican)
            .map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.name}
              </option>
            ))}
        </optgroup>
      </select>
    </div>
  );
}; 